//===--- PassByValueCheck.cpp - clang-tidy---------------------------------===//
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
        if (TypeSourceInfo *FriendTypeSource = FriendDecl->getFriendType()) {
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
        for (const CXXConstructorDecl *Ctor : Node.ctors()) {
          if (Ctor->isMoveConstructor() && !Ctor->isDeleted() &&
              (Ctor->getAccess() == AS_public ||
               (BoundClass && isFirstFriendOfSecond(BoundClass, &Node))))
            return false;
        }
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

/// Whether or not \p ParamDecl is used exactly one time in \p Ctor.
///
/// Checks both in the init-list and the body of the constructor.
static bool paramReferredExactlyOnce(const CXXConstructorDecl *Ctor,
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
    /// the
    /// given constructor.
    bool hasExactlyOneUsageIn(const CXXConstructorDecl *Ctor) {
      Count = 0U;
      TraverseDecl(const_cast<CXXConstructorDecl *>(Ctor));
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

  return ExactlyOneUsageVisitor(ParamDecl).hasExactlyOneUsageIn(Ctor);
}

/// Returns true if the given constructor is part of a lvalue/rvalue reference
/// pair, i.e. `Param` is of lvalue reference type, and there exists another
/// constructor such that:
///  - it has the same number of parameters as `Ctor`.
///  - the parameter at the same index as `Param` is an rvalue reference
///    of the same pointee type
///  - all other parameters have the same type as the corresponding parameter in
///    `Ctor` or are rvalue references with the same pointee type.
/// Examples:
///  A::A(const B& Param)
///  A::A(B&&)
///
///  A::A(const B& Param, const C&)
///  A::A(B&& Param, C&&)
///
///  A::A(const B&, const C& Param)
///  A::A(B&&, C&& Param)
///
///  A::A(const B&, const C& Param)
///  A::A(const B&, C&& Param)
///
///  A::A(const B& Param, int)
///  A::A(B&& Param, int)
static bool hasRValueOverload(const CXXConstructorDecl *Ctor,
                              const ParmVarDecl *Param) {
  if (!Param->getType().getCanonicalType()->isLValueReferenceType()) {
    // The parameter is passed by value.
    return false;
  }
  const int ParamIdx = Param->getFunctionScopeIndex();
  const CXXRecordDecl *Record = Ctor->getParent();

  // Check whether a ctor `C` forms a pair with `Ctor` under the aforementioned
  // rules.
  const auto IsRValueOverload = [&Ctor, ParamIdx](const CXXConstructorDecl *C) {
    if (C == Ctor || C->isDeleted() ||
        C->getNumParams() != Ctor->getNumParams())
      return false;
    for (int I = 0, E = C->getNumParams(); I < E; ++I) {
      const clang::QualType CandidateParamType =
          C->parameters()[I]->getType().getCanonicalType();
      const clang::QualType CtorParamType =
          Ctor->parameters()[I]->getType().getCanonicalType();
      const bool IsLValueRValuePair =
          CtorParamType->isLValueReferenceType() &&
          CandidateParamType->isRValueReferenceType() &&
          CandidateParamType->getPointeeType()->getUnqualifiedDesugaredType() ==
              CtorParamType->getPointeeType()->getUnqualifiedDesugaredType();
      if (I == ParamIdx) {
        // The parameter of interest must be paired.
        if (!IsLValueRValuePair)
          return false;
      } else {
        // All other parameters can be similar or paired.
        if (!(CandidateParamType == CtorParamType || IsLValueRValuePair))
          return false;
      }
    }
    return true;
  };

  for (const auto *Candidate : Record->ctors()) {
    if (IsRValueOverload(Candidate))
      return true;
  }
  return false;
}

/// Find all references to \p ParamDecl across all of the
/// redeclarations of \p Ctor.
static SmallVector<const ParmVarDecl *, 2>
collectParamDecls(const CXXConstructorDecl *Ctor,
                  const ParmVarDecl *ParamDecl) {
  SmallVector<const ParmVarDecl *, 2> Results;
  unsigned ParamIdx = ParamDecl->getFunctionScopeIndex();

  for (const FunctionDecl *Redecl : Ctor->redecls())
    Results.push_back(Redecl->getParamDecl(ParamIdx));
  return Results;
}

PassByValueCheck::PassByValueCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      Inserter(Options.getLocalOrGlobal("IncludeStyle",
                                        utils::IncludeSorter::IS_LLVM),
               areDiagsSelfContained()),
      ValuesOnly(Options.get("ValuesOnly", false)) {}

void PassByValueCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", Inserter.getStyle());
  Options.store(Opts, "ValuesOnly", ValuesOnly);
}

void PassByValueCheck::registerMatchers(MatchFinder *Finder) {
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
}

void PassByValueCheck::registerPPCallbacks(const SourceManager &SM,
                                           Preprocessor *PP,
                                           Preprocessor *ModuleExpanderPP) {
  Inserter.registerPreprocessor(PP);
}

void PassByValueCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructorDecl>("Ctor");
  const auto *ParamDecl = Result.Nodes.getNodeAs<ParmVarDecl>("Param");
  const auto *Initializer =
      Result.Nodes.getNodeAs<CXXCtorInitializer>("Initializer");
  SourceManager &SM = *Result.SourceManager;

  // If the parameter is used or anything other than the copy, do not apply
  // the changes.
  if (!paramReferredExactlyOnce(Ctor, ParamDecl))
    return;

  // If the parameter is trivial to copy, don't move it. Moving a trivially
  // copyable type will cause a problem with performance-move-const-arg
  if (ParamDecl->getType().getNonReferenceType().isTriviallyCopyableType(
          *Result.Context))
    return;

  // Do not trigger if we find a paired constructor with an rvalue.
  if (hasRValueOverload(Ctor, ParamDecl))
    return;

  auto Diag = diag(ParamDecl->getBeginLoc(), "pass by value and use std::move");

  // If we received a `const&` type, we need to rewrite the function
  // declarations.
  if (ParamDecl->getType()->isLValueReferenceType()) {
    // Check if we can succesfully rewrite all declarations of the constructor.
    for (const ParmVarDecl *ParmDecl : collectParamDecls(Ctor, ParamDecl)) {
      TypeLoc ParamTL = ParmDecl->getTypeSourceInfo()->getTypeLoc();
      auto RefTL = ParamTL.getAs<ReferenceTypeLoc>();
      if (RefTL.isNull()) {
        // We cannot rewrite this instance. The type is probably hidden behind
        // some `typedef`. Do not offer a fix-it in this case.
        return;
      }
    }
    // Rewrite all declarations.
    for (const ParmVarDecl *ParmDecl : collectParamDecls(Ctor, ParamDecl)) {
      TypeLoc ParamTL = ParmDecl->getTypeSourceInfo()->getTypeLoc();
      auto RefTL = ParamTL.getAs<ReferenceTypeLoc>();

      TypeLoc ValueTL = RefTL.getPointeeLoc();
      CharSourceRange TypeRange = CharSourceRange::getTokenRange(
          ParmDecl->getBeginLoc(), ParamTL.getEndLoc());
      std::string ValueStr =
          Lexer::getSourceText(
              CharSourceRange::getTokenRange(ValueTL.getSourceRange()), SM,
              getLangOpts())
              .str();
      ValueStr += ' ';
      Diag << FixItHint::CreateReplacement(TypeRange, ValueStr);
    }
  }

  // Use std::move in the initialization list.
  Diag << FixItHint::CreateInsertion(Initializer->getRParenLoc(), ")")
       << FixItHint::CreateInsertion(
              Initializer->getLParenLoc().getLocWithOffset(1), "std::move(")
       << Inserter.createIncludeInsertion(
              Result.SourceManager->getFileID(Initializer->getSourceLocation()),
              "<utility>");
}

} // namespace clang::tidy::modernize
