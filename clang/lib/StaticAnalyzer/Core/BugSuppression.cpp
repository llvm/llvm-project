//===- BugSuppression.cpp - Suppression interface -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/BugReporter/BugSuppression.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugReporter.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/TimeProfiler.h"

using namespace clang;
using namespace ento;

namespace {

using Ranges = llvm::SmallVectorImpl<SourceRange>;

inline bool hasSuppression(const Decl *D) {
  // FIXME: Implement diagnostic identifier arguments
  // (checker names, "hashtags").
  if (const auto *Suppression = D->getAttr<SuppressAttr>())
    return !Suppression->isGSL() &&
           (Suppression->diagnosticIdentifiers().empty());
  return false;
}
inline bool hasSuppression(const AttributedStmt *S) {
  // FIXME: Implement diagnostic identifier arguments
  // (checker names, "hashtags").
  return llvm::any_of(S->getAttrs(), [](const Attr *A) {
    const auto *Suppression = dyn_cast<SuppressAttr>(A);
    return Suppression && !Suppression->isGSL() &&
           (Suppression->diagnosticIdentifiers().empty());
  });
}

template <class NodeType> inline SourceRange getRange(const NodeType *Node) {
  return Node->getSourceRange();
}
template <> inline SourceRange getRange(const AttributedStmt *S) {
  // Begin location for attributed statement node seems to be ALWAYS invalid.
  //
  // It is unlikely that we ever report any warnings on suppression
  // attribute itself, but even if we do, we wouldn't want that warning
  // to be suppressed by that same attribute.
  //
  // Long story short, we can use inner statement and it's not going to break
  // anything.
  return getRange(S->getSubStmt());
}

inline bool isLessOrEqual(SourceLocation LHS, SourceLocation RHS,
                          const SourceManager &SM) {
  // SourceManager::isBeforeInTranslationUnit tests for strict
  // inequality, when we need a non-strict comparison (bug
  // can be reported directly on the annotated note).
  // For this reason, we use the following equivalence:
  //
  //   A <= B <==> !(B < A)
  //
  return !SM.isBeforeInTranslationUnit(RHS, LHS);
}

inline bool fullyContains(SourceRange Larger, SourceRange Smaller,
                          const SourceManager &SM) {
  // Essentially this means:
  //
  //   Larger.fullyContains(Smaller)
  //
  // However, that method has a very trivial implementation and couldn't
  // compare regular locations and locations from macro expansions.
  // We could've converted everything into regular locations as a solution,
  // but the following solution seems to be the most bulletproof.
  return isLessOrEqual(Larger.getBegin(), Smaller.getBegin(), SM) &&
         isLessOrEqual(Smaller.getEnd(), Larger.getEnd(), SM);
}

class CacheInitializer : public DynamicRecursiveASTVisitor {
public:
  static void initialize(const Decl *D, Ranges &ToInit) {
    CacheInitializer(ToInit).TraverseDecl(const_cast<Decl *>(D));
  }

  bool VisitDecl(Decl *D) override {
    // Bug location could be somewhere in the init value of
    // a freshly declared variable.  Even though it looks like the
    // user applied attribute to a statement, it will apply to a
    // variable declaration, and this is where we check for it.
    return VisitAttributedNode(D);
  }

  bool VisitAttributedStmt(AttributedStmt *AS) override {
    // When we apply attributes to statements, it actually creates
    // a wrapper statement that only contains attributes and the wrapped
    // statement.
    return VisitAttributedNode(AS);
  }

private:
  template <class NodeType> bool VisitAttributedNode(NodeType *Node) {
    if (hasSuppression(Node)) {
      // TODO: In the future, when we come up with good stable IDs for checkers
      //       we can return a list of kinds to ignore, or all if no arguments
      //       were provided.
      addRange(getRange(Node));
    }
    // We should keep traversing AST.
    return true;
  }

  void addRange(SourceRange R) {
    if (R.isValid()) {
      Result.push_back(R);
    }
  }

  CacheInitializer(Ranges &R) : Result(R) {
    ShouldVisitTemplateInstantiations = true;
    ShouldWalkTypesOfTypeLocs = false;
    ShouldVisitImplicitCode = false;
    ShouldVisitLambdaBody = true;
  }
  Ranges &Result;
};

std::string timeScopeName(const Decl *DeclWithIssue) {
  if (!llvm::timeTraceProfilerEnabled())
    return "";
  return llvm::formatv(
             "BugSuppression::isSuppressed init suppressions cache for {0}",
             DeclWithIssue->getDeclKindName())
      .str();
}

llvm::TimeTraceMetadata getDeclTimeTraceMetadata(const Decl *DeclWithIssue) {
  assert(DeclWithIssue);
  assert(llvm::timeTraceProfilerEnabled());
  std::string Name = "<noname>";
  if (const auto *ND = dyn_cast<NamedDecl>(DeclWithIssue)) {
    Name = ND->getNameAsString();
  }
  const auto &SM = DeclWithIssue->getASTContext().getSourceManager();
  auto Line = SM.getPresumedLineNumber(DeclWithIssue->getBeginLoc());
  auto Fname = SM.getFilename(DeclWithIssue->getBeginLoc());
  return llvm::TimeTraceMetadata{std::move(Name), Fname.str(),
                                 static_cast<int>(Line)};
}

} // end anonymous namespace

// TODO: Introduce stable IDs for checkers and check for those here
//       to be more specific.  Attribute without arguments should still
//       be considered as "suppress all".
//       It is already much finer granularity than what we have now
//       (i.e. removing the whole function from the analysis).
bool BugSuppression::isSuppressed(const BugReport &R) {
  PathDiagnosticLocation Location = R.getLocation();
  PathDiagnosticLocation UniqueingLocation = R.getUniqueingLocation();
  const Decl *DeclWithIssue = R.getDeclWithIssue();

  return isSuppressed(Location, DeclWithIssue, {}) ||
         isSuppressed(UniqueingLocation, DeclWithIssue, {});
}

static const ClassTemplateDecl *
walkInstantiatedFromChain(const ClassTemplateDecl *Tmpl) {
  // For nested member templates (e.g., S2 inside S1<T>), getInstantiatedFrom
  // may return the member template as instantiated within an outer
  // specialization (e.g., S2 as it appears in S1<int>).  That instantiated
  // member template has no definition redeclaration itself; we need to walk
  // up the member template chain to reach the primary template definition.
  // \code
  //   template <class> struct S1 {
  //     template <class> struct S2 {
  //       int i;
  //       template <class T> int m(const S2<T>& s2) {
  //         return s2.i;
  //       }
  //     };
  //   }
  // /code
  const ClassTemplateDecl *MemberTmpl;
  while ((MemberTmpl = Tmpl->getInstantiatedFromMemberTemplate())) {
    if (Tmpl->isMemberSpecialization())
      break;
    Tmpl = MemberTmpl;
  }
  return Tmpl;
}

static const ClassTemplatePartialSpecializationDecl *walkInstantiatedFromChain(
    const ClassTemplatePartialSpecializationDecl *PartialSpec) {
  const ClassTemplatePartialSpecializationDecl *MemberPS;
  while ((MemberPS = PartialSpec->getInstantiatedFromMember())) {
    if (PartialSpec->isMemberSpecialization())
      break;
    PartialSpec = MemberPS;
  }
  return PartialSpec;
}

template <class T> static const T *chooseDefinitionRedecl(const T *Tmpl) {
  static_assert(llvm::is_one_of<T, ClassTemplateDecl,
                                ClassTemplatePartialSpecializationDecl>::value);
  for (const auto *Redecl : Tmpl->redecls()) {
    if (const T *D = cast<T>(Redecl); D->isThisDeclarationADefinition()) {
      return D;
    }
  }
  assert(false && "This template must have a redecl that is a definition");
  return Tmpl;
}

// For template specializations, returns the primary template definition or
// partial specialization that was used to instantiate the specialization.
// This ensures suppression attributes on templates apply to their
// specializations.
//
// For example, given:
//   template <typename T> class [[clang::suppress]] Wrapper { ... };
//   Wrapper<int> w; // instantiates ClassTemplateSpecializationDecl
//
// When analyzing code in Wrapper<int>, this function maps the specialization
// back to the primary template definition, allowing us to find the suppression
// attribute.
//
// The function handles specializations (and partial specializations) of
// class and function templates.
// For any other decl, it returns the input unchagned.
static const Decl *
preferTemplateDefinitionForTemplateSpecializations(const Decl *D) {
  // For function template specializations (including instantiated friend
  // function templates), map back to the primary template's FunctionDecl so
  // that the lexical parent chain walk reaches the class where the template
  // was defined inline.
  //
  // This handles the case where a friend function template is defined inline
  // inside a [[clang::suppress]]-annotated class but was pre-declared at
  // namespace scope.  In that case the instantiation's lexical DC is the
  // namespace (from the pre-declaration), not the class.  Walking back to the
  // primary template FunctionDecl — whose lexical DC IS the class — lets the
  // existing parent-chain walk find the suppression attribute.
  if (const auto *FD = dyn_cast<FunctionDecl>(D)) {
    if (const FunctionDecl *Pattern = FD->getTemplateInstantiationPattern())
      return Pattern;
  }

  const auto *SpecializationDecl = dyn_cast<ClassTemplateSpecializationDecl>(D);
  if (!SpecializationDecl)
    return D;

  auto InstantiatedFrom = SpecializationDecl->getInstantiatedFrom();
  if (!InstantiatedFrom)
    return D;

  if (const auto *Tmpl = InstantiatedFrom.dyn_cast<ClassTemplateDecl *>()) {
    // Interestingly, the source template might be a forward declaration, so we
    // need to find the definition redeclaration.
    return chooseDefinitionRedecl(walkInstantiatedFromChain(Tmpl));
  }
  return chooseDefinitionRedecl(walkInstantiatedFromChain(
      cast<ClassTemplatePartialSpecializationDecl *>(InstantiatedFrom)));
}

bool BugSuppression::isSuppressed(const PathDiagnosticLocation &Location,
                                  const Decl *DeclWithIssue,
                                  DiagnosticIdentifierList Hashtags) {
  if (!Location.isValid())
    return false;

  if (!DeclWithIssue) {
    // FIXME: This defeats the purpose of passing DeclWithIssue to begin with.
    // If this branch is ever hit, we're re-doing all the work we've already
    // done as well as perform a lot of work we'll never need.
    // Gladly, none of our on-by-default checkers currently need it.
    DeclWithIssue = ACtx.getTranslationUnitDecl();
  } else {
    // This is the fast path. However, we should still consider the topmost
    // declaration that isn't TranslationUnitDecl, because we should respect
    // attributes on the entire declaration chain.
    while (true) {

      // Template specializations (e.g., Wrapper<int>) should inherit
      // suppression attributes from their primary template or partial
      // specialization. Transform specializations to their template definitions
      // before checking for suppressions or walking up the lexical parent
      // chain.
      // Simply taking the lexical parent of template specializations might land
      // us in a completely different namespace.
      DeclWithIssue =
          preferTemplateDefinitionForTemplateSpecializations(DeclWithIssue);

      // Use the "lexical" parent. Eg., if the attribute is on a class, suppress
      // warnings in inline methods but not in out-of-line methods.
      const Decl *Parent =
          dyn_cast_or_null<Decl>(DeclWithIssue->getLexicalDeclContext());
      if (Parent == nullptr || isa<TranslationUnitDecl>(Parent))
        break;

      DeclWithIssue = Parent;
    }
  }

  // While some warnings are attached to AST nodes (mostly path-sensitive
  // checks), others are simply associated with a plain source location
  // or range.  Figuring out the node based on locations can be tricky,
  // so instead, we traverse the whole body of the declaration and gather
  // information on ALL suppressions.  After that we can simply check if
  // any of those suppressions affect the warning in question.
  //
  // Traversing AST of a function is not a heavy operation, but for
  // large functions with a lot of bugs it can make a dent in performance.
  // In order to avoid this scenario, we cache traversal results.
  auto InsertionResult = CachedSuppressionLocations.insert(
      std::make_pair(DeclWithIssue, CachedRanges{}));
  Ranges &SuppressionRanges = InsertionResult.first->second;
  if (InsertionResult.second) {
    llvm::TimeTraceScope TimeScope(
        timeScopeName(DeclWithIssue),
        [DeclWithIssue]() { return getDeclTimeTraceMetadata(DeclWithIssue); });
    // We haven't checked this declaration for suppressions yet!
    CacheInitializer::initialize(DeclWithIssue, SuppressionRanges);
  }

  SourceRange BugRange = Location.asRange();
  const SourceManager &SM = Location.getManager();

  return llvm::any_of(SuppressionRanges,
                      [BugRange, &SM](SourceRange Suppression) {
                        return fullyContains(Suppression, BugRange, SM);
                      });
}
