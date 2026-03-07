//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassByValueCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Lex/Lexer.h"
#include "clang/Lex/Preprocessor.h"

using namespace clang::ast_matchers;
using namespace llvm;

namespace clang::tidy::modernize {

static bool isFirstFriendOfSecond(const CXXRecordDecl *Friend,
                                  const CXXRecordDecl *Class) {
  return llvm::any_of(
      Class->friends(), [Friend](FriendDecl *FriendDecl) -> bool {
        if (const TypeSourceInfo *FriendTypeSource =
                FriendDecl->getFriendType()) {
          const QualType FriendType = FriendTypeSource->getType();
          return FriendType->getAsCXXRecordDecl() == Friend;
        }
        return false;
      });
}

namespace {
/// Matches move-constructible classes whose constructor can be called inside
/// a CXXRecordDecl with a bound ID.
///
/// Given
/// \code
///   // POD types are trivially move constructible.
///   struct Foo { int a; };
///
///   struct Bar {
///     Bar(Bar &&) = deleted;
///     int a;
///   };
///
///   class Buz {
///     Buz(Buz &&);
///     int a;
///     friend class Outer;
///   };
///
///   class Outer {
///   };
/// \endcode
/// recordDecl(isMoveConstructibleInBoundCXXRecordDecl("Outer"))
///   matches "Foo", "Buz".
AST_MATCHER_P(CXXRecordDecl, isMoveConstructibleInBoundCXXRecordDecl, StringRef,
              RecordDeclID) {
  return Builder->removeBindings(
      [this,
       &Node](const ast_matchers::internal::BoundNodesMap &Nodes) -> bool {
        const auto *BoundClass =
            Nodes.getNode(this->RecordDeclID).get<CXXRecordDecl>();
        for (const CXXConstructorDecl *Ctor : Node.ctors())
          if (Ctor->isMoveConstructor() && !Ctor->isDeleted() &&
              (Ctor->getAccess() == AS_public ||
               (BoundClass && isFirstFriendOfSecond(BoundClass, &Node))))
            return false;
        return true;
      });
}
} // namespace

static TypeMatcher notTemplateSpecConstRefType() {
  return lValueReferenceType(
      pointee(unless(templateSpecializationType()), isConstQualified()));
}

static TypeMatcher nonConstValueType() {
  return qualType(unless(anyOf(referenceType(), isConstQualified())));
}

/// Whether or not \p ParamDecl is used exactly one time in \p Func.
///
/// Checks both in the init-list (for constructors) and the body of the
/// function.
static bool paramReferredExactlyOnce(const FunctionDecl *Func,
                                     const ParmVarDecl *ParamDecl) {
  /// \c clang::RecursiveASTVisitor that checks that the given
  /// \c ParmVarDecl is used exactly one time.
  ///
  /// \see ExactlyOneUsageVisitor::hasExactlyOneUsageIn()
  class ExactlyOneUsageVisitor
      : public RecursiveASTVisitor<ExactlyOneUsageVisitor> {
    friend class RecursiveASTVisitor<ExactlyOneUsageVisitor>;

  public:
    ExactlyOneUsageVisitor(const ParmVarDecl *ParamDecl)
        : ParamDecl(ParamDecl) {}

    /// Whether or not the parameter variable is referred only once in
    /// the given function.
    bool hasExactlyOneUsageIn(const FunctionDecl *Func) {
      Count = 0U;
      TraverseDecl(const_cast<FunctionDecl *>(Func));
      return Count == 1U;
    }

  private:
    /// Counts the number of references to a variable.
    ///
    /// Stops the AST traversal if more than one usage is found.
    bool VisitDeclRefExpr(DeclRefExpr *D) {
      if (const ParmVarDecl *To = dyn_cast<ParmVarDecl>(D->getDecl())) {
        if (To == ParamDecl) {
          ++Count;
          if (Count > 1U) {
            // No need to look further, used more than once.
            return false;
          }
        }
      }
      return true;
    }

    const ParmVarDecl *ParamDecl;
    unsigned Count = 0U;
  };

  return ExactlyOneUsageVisitor(ParamDecl).hasExactlyOneUsageIn(Func);
}

/// Returns true if \p Func has a paired overload where the parameter at the
/// same index as \p Param is an rvalue reference of the same pointee type.
///
/// For constructors, only other constructors of the same class are checked.
/// For free/member functions, overloads are looked up in the same
/// DeclContext (namespace or class).
///
/// Examples:
///  void foo(const B& Param)
///  void foo(B&&)
///
///  A::A(const B& Param)
///  A::A(B&&)
static bool hasRValueOverload(const FunctionDecl *Func,
                              const ParmVarDecl *Param) {
  if (!Param->getType().getCanonicalType()->isLValueReferenceType()) {
    // The parameter is passed by value.
    return false;
  }
  const int ParamIdx = Param->getFunctionScopeIndex();

  // Helper to check whether a candidate function forms an lvalue/rvalue pair.
  const auto IsRValueOverload = [Func,
                                 ParamIdx](const FunctionDecl *Candidate) {
    if (Candidate == Func || Candidate->isDeleted() ||
        Candidate->getNumParams() != Func->getNumParams())
      return false;
    for (int I = 0, E = Candidate->getNumParams(); I < E; ++I) {
      const clang::QualType CandidateParamType =
          Candidate->parameters()[I]->getType().getCanonicalType();
      const clang::QualType FuncParamType =
          Func->parameters()[I]->getType().getCanonicalType();
      const bool IsLValueRValuePair =
          FuncParamType->isLValueReferenceType() &&
          CandidateParamType->isRValueReferenceType() &&
          CandidateParamType->getPointeeType()->getUnqualifiedDesugaredType() ==
              FuncParamType->getPointeeType()->getUnqualifiedDesugaredType();
      if (I == ParamIdx) {
        // The parameter of interest must be paired.
        if (!IsLValueRValuePair)
          return false;
      } else {
        // All other parameters can be similar or paired.
        if (!(CandidateParamType == FuncParamType || IsLValueRValuePair))
          return false;
      }
    }
    return true;
  };

  // For constructors, check sibling constructors.
  if (const auto *Ctor = dyn_cast<CXXConstructorDecl>(Func)) {
    const CXXRecordDecl *Record = Ctor->getParent();
    return llvm::any_of(Record->ctors(), IsRValueOverload);
  }

  // For other functions, look up overloads in the same DeclContext.
  const DeclContext *DC = Func->getDeclContext();
  const DeclarationName Name = Func->getDeclName();
  return llvm::any_of(DC->lookup(Name), [IsRValueOverload](const Decl *D) {
    const auto *FD = dyn_cast<FunctionDecl>(D);
    return FD && IsRValueOverload(FD);
  });
}

/// Find all references to \p ParamDecl across all of the
/// redeclarations of \p Func.
static SmallVector<const ParmVarDecl *, 2>
collectParamDecls(const FunctionDecl *Func, const ParmVarDecl *ParamDecl) {
  SmallVector<const ParmVarDecl *, 2> Results;
  const unsigned ParamIdx = ParamDecl->getFunctionScopeIndex();

  for (const FunctionDecl *Redecl : Func->redecls())
    Results.push_back(Redecl->getParamDecl(ParamIdx));
  return Results;
}

PassByValueCheck::PassByValueCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      ValuesOnly(Options.get("ValuesOnly", false)),
      IgnoreMacros(Options.get("IgnoreMacros", false)) {}

void PassByValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "ValuesOnly", ValuesOnly);
  Options.store(Opts, "IgnoreMacros", IgnoreMacros);
}

void PassByValueCheck::registerMatchers(MatchFinder *Finder) {
  // Matcher for constructor member initializer lists (existing behavior).
  Finder->addMatcher(
      traverse(
          TK_AsIs,
          cxxConstructorDecl(
              ofClass(cxxRecordDecl().bind("outer")),
              forEachConstructorInitializer(
                  cxxCtorInitializer(
                      unless(isBaseInitializer()),
                      // Clang builds a CXXConstructExpr only when it knows
                      // which constructor will be called. In dependent contexts
                      // a ParenListExpr is generated instead of a
                      // CXXConstructExpr, filtering out templates automatically
                      // for us.
                      withInitializer(cxxConstructExpr(
                          has(ignoringParenImpCasts(declRefExpr(to(
                              parmVarDecl(
                                  hasType(qualType(
                                      // Match only const-ref or a non-const
                                      // value parameters. Rvalues,
                                      // TemplateSpecializationValues and
                                      // const-values shouldn't be modified.
                                      ValuesOnly
                                          ? nonConstValueType()
                                          : anyOf(notTemplateSpecConstRefType(),
                                                  nonConstValueType()))))
                                  .bind("Param"))))),
                          hasDeclaration(cxxConstructorDecl(
                              isCopyConstructor(), unless(isDeleted()),
                              hasDeclContext(cxxRecordDecl(
                                  isMoveConstructibleInBoundCXXRecordDecl(
                                      "outer"))))))))
                      .bind("Initializer")))
              .bind("Ctor")),
      this);

  // Matcher for function body local variable copies from const-ref parameters.
  // Matches patterns like:
  //   void f(const T& param) { T local = param; }
  if (!ValuesOnly) {
    Finder->addMatcher(
        traverse(
            TK_AsIs,
            functionDecl(
                unless(isInstantiated()),
                hasBody(hasDescendant(
                    varDecl(
                        hasLocalStorage(),
                        hasInitializer(ignoringImplicit(cxxConstructExpr(
                            hasDeclaration(cxxConstructorDecl(
                                isCopyConstructor(), unless(isDeleted()))),
                            hasArgument(
                                0,
                                ignoringImplicit(declRefExpr(to(
                                    parmVarDecl(
                                        hasType(notTemplateSpecConstRefType()))
                                        .bind("FuncParam")))))))))
                        .bind("LocalVar"))))
                .bind("Func")),
        this);
  }
}

void PassByValueCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

/// Attempts to rewrite the const-ref parameter declarations to pass-by-value
/// across all redeclarations. Returns true if fixits were added, false if
/// rewriting is not possible (e.g. type hidden behind a typedef).
static bool rewriteParamDeclsToValue(const FunctionDecl *Func,
                                     const ParmVarDecl *ParamDecl,
                                     const SourceManager &SM,
                                     const LangOptions &LangOpts,
                                     DiagnosticBuilder &Diag) {
  if (!ParamDecl->getType()->isLValueReferenceType())
    return true; // Already by value, nothing to rewrite.

  // Check if we can successfully rewrite all declarations.
  for (const ParmVarDecl *ParmDecl : collectParamDecls(Func, ParamDecl)) {
    const TypeLoc ParamTL = ParmDecl->getTypeSourceInfo()->getTypeLoc();
    auto RefTL = ParamTL.getAs<ReferenceTypeLoc>();
    if (RefTL.isNull()) {
      // We cannot rewrite this instance. The type is probably hidden behind
      // some `typedef`. Do not offer a fix-it in this case.
      return false;
    }
  }
  // Rewrite all declarations.
  for (const ParmVarDecl *ParmDecl : collectParamDecls(Func, ParamDecl)) {
    const TypeLoc ParamTL = ParmDecl->getTypeSourceInfo()->getTypeLoc();
    auto RefTL = ParamTL.getAs<ReferenceTypeLoc>();

    const TypeLoc ValueTL = RefTL.getPointeeLoc();
    const CharSourceRange TypeRange = CharSourceRange::getTokenRange(
        ParmDecl->getBeginLoc(), ParamTL.getEndLoc());
    std::string ValueStr = Lexer::getSourceText(CharSourceRange::getTokenRange(
                                                    ValueTL.getSourceRange()),
                                                SM, LangOpts)
                               .str();
    ValueStr += ' ';
    Diag << FixItHint::CreateReplacement(TypeRange, ValueStr);
  }
  return true;
}

void PassByValueCheck::check(const MatchFinder::MatchResult &Result) {
  const SourceManager &SM = *Result.SourceManager;

  // Case 1: Constructor member initializer list.
  if (const auto *Initializer =
          Result.Nodes.getNodeAs<CXXCtorInitializer>("Initializer")) {
    const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("Ctor");
    const auto *ParamDecl = Result.Nodes.getNodeAs<ParmVarDecl>("Param");

    if (IgnoreMacros && ParamDecl->getBeginLoc().isMacroID())
      return;

    if (!paramReferredExactlyOnce(Ctor, ParamDecl))
      return;

    if (ParamDecl->getType().getNonReferenceType().isTriviallyCopyableType(
            *Result.Context))
      return;

    if (hasRValueOverload(Ctor, ParamDecl))
      return;

    auto Diag =
        diag(ParamDecl->getBeginLoc(), "pass by value and use std::move");

    if (!rewriteParamDeclsToValue(Ctor, ParamDecl, SM, getLangOpts(), Diag))
      return;

    // Use std::move in the initialization list.
    Diag << FixItHint::CreateInsertion(Initializer->getRParenLoc(), ")")
         << FixItHint::CreateInsertion(
                Initializer->getLParenLoc().getLocWithOffset(1), "std::move(")
         << Inserter.createIncludeInsertion(
                Result.SourceManager->getFileID(
                    Initializer->getSourceLocation()),
                "<utility>");
    return;
  }

  // Case 2: Function body local variable copy.
  const auto *Func = Result.Nodes.getNodeAs<FunctionDecl>("Func");
  const auto *ParamDecl = Result.Nodes.getNodeAs<ParmVarDecl>("FuncParam");
  const auto *LocalVar = Result.Nodes.getNodeAs<VarDecl>("LocalVar");
  if (!Func || !ParamDecl || !LocalVar)
    return;

  if (IgnoreMacros && ParamDecl->getBeginLoc().isMacroID())
    return;

  if (!paramReferredExactlyOnce(Func, ParamDecl))
    return;

  if (ParamDecl->getType().getNonReferenceType().isTriviallyCopyableType(
          *Result.Context))
    return;

  if (hasRValueOverload(Func, ParamDecl))
    return;

  // Check that the copied type has a usable move constructor.
  const QualType LocalType = LocalVar->getType().getCanonicalType();
  const CXXRecordDecl *Record = LocalType->getAsCXXRecordDecl();
  if (!Record || !Record->hasDefinition())
    return;
  bool HasUsableMove = false;
  for (const CXXConstructorDecl *Ctor : Record->ctors())
    if (Ctor->isMoveConstructor() && !Ctor->isDeleted()) {
      HasUsableMove = true;
      break;
    }
  if (!HasUsableMove && !Record->needsImplicitMoveConstructor())
    return;

  auto Diag = diag(ParamDecl->getBeginLoc(), "pass by value and use std::move");

  if (!rewriteParamDeclsToValue(Func, ParamDecl, SM, getLangOpts(), Diag))
    return;

  // Wrap the initializer with std::move().
  const Expr *Init = LocalVar->getInit()->IgnoreImplicit();
  if (const auto *Construct = dyn_cast<CXXConstructExpr>(Init)) {
    if (Construct->getNumArgs() > 0) {
      const Expr *Arg = Construct->getArg(0)->IgnoreImplicit();
      Diag << FixItHint::CreateInsertion(Arg->getBeginLoc(), "std::move(")
           << FixItHint::CreateInsertion(
                  Lexer::getLocForEndOfToken(Arg->getEndLoc(), 0, SM,
                                             getLangOpts()),
                  ")");
    }
  }

  Diag << Inserter.createIncludeInsertion(
      Result.SourceManager->getFileID(LocalVar->getLocation()), "<utility>");
}

} // namespace clang::tidy::modernize
