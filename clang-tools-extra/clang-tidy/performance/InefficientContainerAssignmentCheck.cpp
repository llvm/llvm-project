//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "InefficientContainerAssignmentCheck.h"
#include "../utils/OptionsUtils.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/ParentMapContext.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include <optional>
#include <string>

using namespace clang::ast_matchers;

namespace clang::tidy::performance {

namespace {

/// How the assignment can be rewritten without the temporary. The order
/// matches the %select in the diagnostic.
enum class Rewrite {
  Assign,      ///< lhs.assign(args...)
  AssignRange, ///< lhs.assign_range(range)
  ClearResize, ///< lhs.clear(); lhs.resize(count)
  InitList,    ///< lhs = {...}
  Direct,      ///< lhs = source
};

} // namespace

/// Collects the variables and members named by \p E so that arguments
/// referring back to the destination can be detected.
static void
collectReferencedDecls(const Expr &E, ASTContext &Ctx,
                       llvm::SmallPtrSetImpl<const ValueDecl *> &Decls) {
  for (const BoundNodes &N : match(findAll(declRefExpr().bind("ref")), E, Ctx))
    Decls.insert(N.getNodeAs<DeclRefExpr>("ref")->getDecl());
  for (const BoundNodes &N :
       match(findAll(memberExpr().bind("member")), E, Ctx))
    Decls.insert(N.getNodeAs<MemberExpr>("member")->getMemberDecl());
}

static bool
refersToAnyOf(const Expr &E, ASTContext &Ctx,
              const llvm::SmallPtrSetImpl<const ValueDecl *> &Decls) {
  llvm::SmallPtrSet<const ValueDecl *, 8> Referenced;
  collectReferencedDecls(E, Ctx, Referenced);
  return llvm::any_of(
      Referenced, [&Decls](const ValueDecl *D) { return Decls.contains(D); });
}

/// Returns true if one of the first \p NumArgs parameters of \p Ctor has the
/// type of a template argument of the container other than its element type.
/// That is what the allocator parameter of the standard containers looks
/// like, and an explicitly passed allocator has no counterpart in 'assign'.
static bool passesAllocator(const CXXConstructorDecl &Ctor, unsigned NumArgs,
                            const ASTContext &Ctx) {
  const auto *Spec =
      dyn_cast<ClassTemplateSpecializationDecl>(Ctor.getParent());
  if (!Spec)
    return false;
  const TemplateArgumentList &TemplateArgs = Spec->getTemplateArgs();
  for (unsigned I = 0, E = std::min(NumArgs, Ctor.getNumParams()); I < E; ++I) {
    const QualType ParamType =
        Ctor.getParamDecl(I)->getType().getNonReferenceType();
    if (!ParamType->isRecordType())
      continue;
    for (unsigned J = 1; J < TemplateArgs.size(); ++J) {
      const TemplateArgument &Arg = TemplateArgs[J];
      if (Arg.getKind() == TemplateArgument::Type &&
          ASTContext::hasSameUnqualifiedType(Arg.getAsType(), ParamType))
        return true;
    }
  }
  return false;
}

static bool isFromRangeTag(QualType T) {
  const CXXRecordDecl *RD = T.getNonReferenceType()->getAsCXXRecordDecl();
  return RD && RD->isInStdNamespace() && RD->getName() == "from_range_t";
}

/// A postfix expression can take a member access directly; anything else has
/// to be parenthesized first.
static bool needsParens(const Expr &E) {
  const Expr *Inner = E.IgnoreImpCasts();
  if (const auto *Call = dyn_cast<CXXOperatorCallExpr>(Inner))
    return Call->getOperator() != OO_Subscript &&
           Call->getOperator() != OO_Call;
  return !isa<DeclRefExpr, MemberExpr, ArraySubscriptExpr, CallExpr, ParenExpr>(
      Inner);
}

/// Returns the statement that directly contains \p E, looking through the
/// implicit cleanup wrapper and parentheses, or nullptr if \p E is a
/// subexpression or an initializer.
static const Stmt *getEnclosingStatement(const Expr &E, ASTContext &Ctx) {
  DynTypedNodeList Parents = Ctx.getParents(E);
  while (Parents.size() == 1) {
    const auto *S = Parents[0].get<Stmt>();
    if (!S)
      return nullptr;
    if (!isa<ExprWithCleanups, ParenExpr>(S))
      return S;
    Parents = Ctx.getParents(*S);
  }
  return nullptr;
}

InefficientContainerAssignmentCheck::InefficientContainerAssignmentCheck(
    StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      ContainerClasses(utils::options::parseStringList(Options.get(
          "ContainerClasses", "::std::vector;::std::deque;::std::list;"
                              "::std::forward_list;::std::basic_string"))) {}

void InefficientContainerAssignmentCheck::storeOptions(
    ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "ContainerClasses",
                utils::options::serializeStringList(ContainerClasses));
}

void InefficientContainerAssignmentCheck::registerMatchers(
    MatchFinder *Finder) {
  if (ContainerClasses.empty())
    return;

  const auto Container = cxxRecordDecl(hasAnyName(ContainerClasses));
  const auto Temporary =
      cxxConstructExpr(hasDeclaration(cxxConstructorDecl(ofClass(Container))))
          .bind("ctor");

  // Match: lhs = Container(...) and lhs = Container{...}, where the assignment
  // is the container's own 'operator='. Template instantiations are skipped:
  // the rewrite might not fit every instantiation, and the temporary is only
  // visible where the types are known.
  Finder->addMatcher(
      cxxOperatorCallExpr(
          unless(isInTemplateInstantiation()), hasOverloadedOperatorName("="),
          callee(cxxMethodDecl(ofClass(Container))),
          hasArgument(0, expr().bind("lhs")),
          hasArgument(1, ignoringImplicit(anyOf(
                             Temporary, cxxFunctionalCastExpr(has(
                                            ignoringImplicit(Temporary)))))))
          .bind("op"),
      this);
}

void InefficientContainerAssignmentCheck::check(
    const MatchFinder::MatchResult &Result) {
  const auto *Op = Result.Nodes.getNodeAs<CXXOperatorCallExpr>("op");
  const auto *LHS = Result.Nodes.getNodeAs<Expr>("lhs");
  const auto *Ctor = Result.Nodes.getNodeAs<CXXConstructExpr>("ctor");
  ASTContext &Ctx = *Result.Context;

  // The rewrites call members of the destination, so the temporary must have
  // exactly the destination's type; a derived class or another specialization
  // would go through a different 'operator='.
  if (!ASTContext::hasSameUnqualifiedType(LHS->getType(), Ctor->getType()))
    return;

  const CXXConstructorDecl *CtorDecl = Ctor->getConstructor();
  SmallVector<const Expr *, 4> Args;
  for (const Expr *Arg : Ctor->arguments())
    if (!isa<CXXDefaultArgExpr>(Arg))
      Args.push_back(Arg);

  // 'lhs = Container();' releases the storage rather than reusing it, which is
  // usually the point; 'clear()' would keep the capacity.
  if (Args.empty())
    return;

  // 'assign' has no allocator parameter.
  if (Args.size() >= 2 && passesAllocator(*CtorDecl, Args.size(), Ctx))
    return;

  Rewrite Form = Rewrite::Assign;
  const InitListExpr *InitList = nullptr;
  if (const auto *StdInitList =
          dyn_cast<CXXStdInitializerListExpr>(Args[0]->IgnoreParenImpCasts());
      StdInitList && Args.size() == 1) {
    Form = Rewrite::InitList;
    InitList =
        dyn_cast<InitListExpr>(StdInitList->getSubExpr()->IgnoreImplicit());
  } else if (Args.size() == 1) {
    if (CtorDecl->isCopyOrMoveConstructor()) {
      if (!ASTContext::hasSameUnqualifiedType(Args[0]->getType(),
                                              LHS->getType()))
        return;
      Form = Rewrite::Direct;
    } else if (CtorDecl->getParamDecl(0)
                   ->getType()
                   .getNonReferenceType()
                   ->isIntegralOrEnumerationType()) {
      Form = Rewrite::ClearResize;
    } else {
      // A conversion from another type: the temporary is the conversion
      // itself, and removing it is a different transformation.
      return;
    }
  } else if (isFromRangeTag(CtorDecl->getParamDecl(0)->getType())) {
    if (Args.size() != 2)
      return;
    Form = Rewrite::AssignRange;
  }

  llvm::SmallPtrSet<const ValueDecl *, 4> LHSDecls;
  collectReferencedDecls(*LHS, Ctx, LHSDecls);
  const auto MentionsDestination = [&](const Expr *Arg) {
    return refersToAnyOf(*Arg, Ctx, LHSDecls);
  };

  // 'x = Container(x);' copies a container into itself to trim its capacity;
  // that temporary is the point rather than waste.
  if (Form == Rewrite::Direct && MentionsDestination(Args[0]))
    return;

  // 'assign' must not be given iterators, pointers or elements that refer into
  // the container it replaces. Counts and other arithmetic values are computed
  // before the call and are harmless.
  const bool AliasesDestination = llvm::any_of(Args, [&](const Expr *Arg) {
    return !Arg->getType()->isArithmeticType() && MentionsDestination(Arg);
  });

  // 'assign', 'clear' and 'resize' do not yield the container, so those
  // rewrites need the value of the assignment to be discarded.
  const Stmt *Parent = getEnclosingStatement(*Op, Ctx);
  const bool ValueDiscarded =
      isa_and_nonnull<CompoundStmt, IfStmt, WhileStmt, DoStmt, ForStmt,
                      CXXForRangeStmt, CaseStmt, DefaultStmt, LabelStmt,
                      AttributedStmt>(Parent);
  const bool OwnStatement = isa_and_nonnull<CompoundStmt>(Parent);

  std::optional<std::string> Replacement;
  if (!Op->getBeginLoc().isMacroID() && !Op->getEndLoc().isMacroID()) {
    bool Rewritable = true;
    switch (Form) {
    case Rewrite::Assign:
    case Rewrite::AssignRange:
      Rewritable = ValueDiscarded && !AliasesDestination;
      break;
    case Rewrite::ClearResize:
      // Two statements, in which the destination is named twice and the
      // count is evaluated after 'clear()'.
      Rewritable = OwnStatement && !LHS->HasSideEffects(Ctx) &&
                   !Args[0]->HasSideEffects(Ctx) &&
                   !MentionsDestination(Args[0]);
      break;
    case Rewrite::InitList:
      // The initializer list is materialized before the assignment.
      Rewritable = InitList != nullptr;
      break;
    case Rewrite::Direct:
      break;
    }

    if (Rewritable) {
      const auto GetText = [&](SourceRange R) {
        return Lexer::getSourceText(CharSourceRange::getTokenRange(R),
                                    *Result.SourceManager, getLangOpts());
      };
      const StringRef LHSText = GetText(LHS->getSourceRange());
      bool Valid = !LHSText.empty();
      SmallVector<StringRef, 4> ArgTexts;
      for (const Expr *Arg : Args) {
        ArgTexts.push_back(GetText(Arg->getSourceRange()));
        Valid = Valid && !ArgTexts.back().empty();
      }
      const std::string Dest =
          needsParens(*LHS) ? ("(" + LHSText + ")").str() : LHSText.str();
      if (Valid) {
        switch (Form) {
        case Rewrite::Assign:
          Replacement = Dest + ".assign(" + llvm::join(ArgTexts, ", ") + ")";
          break;
        case Rewrite::AssignRange:
          Replacement = Dest + ".assign_range(" + ArgTexts[1].str() + ")";
          break;
        case Rewrite::ClearResize:
          Replacement =
              Dest + ".clear(); " + Dest + ".resize(" + ArgTexts[0].str() + ")";
          break;
        case Rewrite::InitList: {
          const StringRef ListText = GetText(InitList->getSourceRange());
          if (!ListText.empty())
            Replacement = (LHSText + " = " + ListText).str();
          break;
        }
        case Rewrite::Direct:
          Replacement = (LHSText + " = " + ArgTexts[0]).str();
          break;
        }
      }
    }
  }

  const auto Diag =
      diag(Op->getOperatorLoc(),
           "inefficient assignment from a temporary '%0'; "
           "%select{use 'assign'|use 'assign_range'|use 'clear' and 'resize'|"
           "assign the initializer list directly|assign the source directly}1 "
           "to reuse the existing storage")
      << CtorDecl->getParent()->getQualifiedNameAsString()
      << static_cast<int>(Form);
  if (Replacement)
    Diag << FixItHint::CreateReplacement(Op->getSourceRange(), *Replacement);
}

} // namespace clang::tidy::performance
