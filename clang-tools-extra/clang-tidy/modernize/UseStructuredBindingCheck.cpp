//===--- UseStructuredBindingCheck.cpp - clang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStructuredBindingCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "../utils/OptionsUtils.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {
namespace {
constexpr const char *DefaultPairTypes = "std::pair";
constexpr llvm::StringLiteral PairDeclName = "PairVarD";
constexpr llvm::StringLiteral PairVarTypeName = "PairVarType";
constexpr llvm::StringLiteral FirstVarDeclName = "FirstVarDecl";
constexpr llvm::StringLiteral SecondVarDeclName = "SecondVarDecl";
constexpr llvm::StringLiteral FirstDeclStmtName = "FirstDeclStmt";
constexpr llvm::StringLiteral SecondDeclStmtName = "SecondDeclStmt";
constexpr llvm::StringLiteral FirstTypeName = "FirstType";
constexpr llvm::StringLiteral SecondTypeName = "SecondType";
constexpr llvm::StringLiteral ScopeBlockName = "ScopeBlock";
constexpr llvm::StringLiteral StdTieAssignStmtName = "StdTieAssign";
constexpr llvm::StringLiteral StdTieExprName = "StdTieExpr";
constexpr llvm::StringLiteral ForRangeStmtName = "ForRangeStmt";

/// What qualifiers and specifiers are used to create structured binding
/// declaration, it only supports the following four cases now.
enum TransferType : uint8_t {
  TT_ByVal,
  TT_ByConstVal,
  TT_ByRef,
  TT_ByConstRef
};

/// Try to match exactly two VarDecl inside two DeclStmts, and set binding for
/// the used DeclStmts.
bool matchTwoVarDecl(const DeclStmt *DS1, const DeclStmt *DS2,
                     ast_matchers::internal::Matcher<VarDecl> InnerMatcher1,
                     ast_matchers::internal::Matcher<VarDecl> InnerMatcher2,
                     internal::ASTMatchFinder *Finder,
                     internal::BoundNodesTreeBuilder *Builder) {
  SmallVector<std::pair<const VarDecl *, const DeclStmt *>, 2> Vars;
  auto CollectVarsInDeclStmt = [&Vars](const DeclStmt *DS) -> bool {
    if (!DS)
      return true;

    for (const auto *VD : DS->decls()) {
      if (Vars.size() == 2)
        return false;

      if (const auto *Var = dyn_cast<VarDecl>(VD))
        Vars.emplace_back(Var, DS);
      else
        return false;
    }

    return true;
  };

  if (!CollectVarsInDeclStmt(DS1) || !CollectVarsInDeclStmt(DS2))
    return false;

  if (Vars.size() != 2)
    return false;

  if (InnerMatcher1.matches(*Vars[0].first, Finder, Builder) &&
      InnerMatcher2.matches(*Vars[1].first, Finder, Builder)) {
    Builder->setBinding(FirstDeclStmtName,
                        clang::DynTypedNode::create(*Vars[0].second));
    if (Vars[0].second != Vars[1].second)
      Builder->setBinding(SecondDeclStmtName,
                          clang::DynTypedNode::create(*Vars[1].second));
    return true;
  }

  return false;
}

/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// following two VarDecls matching the inner matcher, at the same time set
/// binding for the CompoundStmt.
AST_MATCHER_P2(Stmt, hasPreTwoVarDecl, ast_matchers::internal::Matcher<VarDecl>,
               InnerMatcher1, ast_matchers::internal::Matcher<VarDecl>,
               InnerMatcher2) {
  DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto I =
      llvm::find(llvm::make_range(C->body_rbegin(), C->body_rend()), &Node);
  assert(I != C->body_rend() && "C is parent of Node");
  if ((I + 1) == C->body_rend())
    return false;

  const auto *DS2 = dyn_cast<DeclStmt>(*(I + 1));
  if (!DS2)
    return false;

  const DeclStmt *DS1 = (!DS2->isSingleDecl() || ((I + 2) == C->body_rend())
                             ? nullptr
                             : dyn_cast<DeclStmt>(*(I + 2)));

  if (matchTwoVarDecl(DS1, DS2, InnerMatcher1, InnerMatcher2, Finder,
                      Builder)) {
    Builder->setBinding(ScopeBlockName, clang::DynTypedNode::create(*C));
    return true;
  }

  return false;
}

/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// followed by two VarDecls matching the inner matcher, at the same time set
/// binding for the CompoundStmt.
AST_MATCHER_P2(Stmt, hasNextTwoVarDecl,
               ast_matchers::internal::Matcher<VarDecl>, InnerMatcher1,
               ast_matchers::internal::Matcher<VarDecl>, InnerMatcher2) {
  DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto *I = llvm::find(C->body(), &Node);
  assert(I != C->body_end() && "C is parent of Node");
  if ((I + 1) == C->body_end())
    return false;

  if (matchTwoVarDecl(
          dyn_cast<DeclStmt>(*(I + 1)),
          ((I + 2) == C->body_end() ? nullptr : dyn_cast<DeclStmt>(*(I + 2))),
          InnerMatcher1, InnerMatcher2, Finder, Builder)) {
    Builder->setBinding(ScopeBlockName, clang::DynTypedNode::create(*C));
    return true;
  }

  return false;
}

/// Matches a Stmt whose parent is a CompoundStmt, and there a two VarDecls
/// matching the inner matcher in the beginning of CompoundStmt.
AST_MATCHER_P2(CompoundStmt, hasFirstTwoVarDecl,
               ast_matchers::internal::Matcher<VarDecl>, InnerMatcher1,
               ast_matchers::internal::Matcher<VarDecl>, InnerMatcher2) {
  const auto *I = Node.body_begin();
  if ((I) == Node.body_end())
    return false;

  return matchTwoVarDecl(
      dyn_cast<DeclStmt>(*(I)),
      ((I + 1) == Node.body_end() ? nullptr : dyn_cast<DeclStmt>(*(I + 1))),
      InnerMatcher1, InnerMatcher2, Finder, Builder);
}

/// It's not very common to have specifiers for variables used to decompose
/// a pair, so we ignore these cases.
AST_MATCHER(VarDecl, hasAnySpecifiersShouldBeIgnored) {
  return Node.isStaticLocal() || Node.isConstexpr() || Node.hasAttrs() ||
         Node.isInlineSpecified();
}

// Ignore nodes inside macros.
AST_POLYMORPHIC_MATCHER(isInMarco,
                        AST_POLYMORPHIC_SUPPORTED_TYPES(Stmt, Decl)) {
  return Node.getBeginLoc().isMacroID() || Node.getEndLoc().isMacroID();
}

AST_MATCHER_P(Expr, ignoringCopyCtorAndImplicitCast,
              ast_matchers::internal::Matcher<Expr>, InnerMatcher) {
  if (const auto *CtorE = dyn_cast<CXXConstructExpr>(&Node)) {
    if (const auto *CtorD = CtorE->getConstructor();
        CtorD->isCopyConstructor() && CtorE->getNumArgs() == 1) {
      return InnerMatcher.matches(*CtorE->getArg(0)->IgnoreImpCasts(), Finder,
                                  Builder);
    }
  }

  return InnerMatcher.matches(*Node.IgnoreImpCasts(), Finder, Builder);
}

} // namespace

UseStructuredBindingCheck::UseStructuredBindingCheck(StringRef Name,
                                                     ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      PairTypes(utils::options::parseStringList(
          Options.get("PairTypes", DefaultPairTypes))) {
  ;
}

static auto getVarInitWithMemberMatcher(StringRef PairName,
                                        StringRef MemberName,
                                        StringRef TypeName,
                                        StringRef BindingName) {
  return varDecl(
             unless(hasAnySpecifiersShouldBeIgnored()), unless(isInMarco()),
             hasInitializer(
                 ignoringImpCasts(ignoringCopyCtorAndImplicitCast(memberExpr(
                     hasObjectExpression(ignoringImpCasts(declRefExpr(
                         to(equalsBoundNode(std::string(PairName)))))),
                     member(fieldDecl(hasName(MemberName),
                                      hasType(qualType().bind(TypeName)))))))))
      .bind(BindingName);
}

void UseStructuredBindingCheck::registerMatchers(MatchFinder *Finder) {
  auto PairType =
      qualType(unless(isVolatileQualified()),
               hasUnqualifiedDesugaredType(recordType(
                   hasDeclaration(cxxRecordDecl(hasAnyName(PairTypes))))));

  auto VarInitWithFirstMember = getVarInitWithMemberMatcher(
      PairDeclName, "first", FirstTypeName, FirstVarDeclName);
  auto VarInitWithSecondMember = getVarInitWithMemberMatcher(
      PairDeclName, "second", SecondTypeName, SecondVarDeclName);

  // X x;
  // Y y;
  // std::tie(x, y) = ...;
  Finder->addMatcher(
      exprWithCleanups(
          unless(isInMarco()),
          has(cxxOperatorCallExpr(
                  hasOverloadedOperatorName("="),
                  hasLHS(ignoringImplicit(
                      callExpr(
                          callee(
                              functionDecl(isInStdNamespace(), hasName("tie"))),
                          hasArgument(
                              0,
                              declRefExpr(to(
                                  varDecl(
                                      unless(hasAnySpecifiersShouldBeIgnored()),
                                      unless(isInMarco()))
                                      .bind(FirstVarDeclName)))),
                          hasArgument(
                              1,
                              declRefExpr(to(
                                  varDecl(
                                      unless(hasAnySpecifiersShouldBeIgnored()),
                                      unless(isInMarco()))
                                      .bind(SecondVarDeclName)))))
                          .bind(StdTieExprName))),
                  hasRHS(expr(hasType(PairType))))
                  .bind(StdTieAssignStmtName)),
          hasPreTwoVarDecl(
              varDecl(equalsBoundNode(std::string(FirstVarDeclName))),
              varDecl(equalsBoundNode(std::string(SecondVarDeclName))))),
      this);

  // pair<X, Y> p = ...;
  // X x = p.first;
  // Y y = p.second;
  Finder->addMatcher(
      declStmt(
          unless(isInMarco()),
          hasSingleDecl(
              varDecl(unless(hasAnySpecifiersShouldBeIgnored()),
                      hasType(qualType(anyOf(PairType, lValueReferenceType(
                                                           pointee(PairType))))
                                  .bind(PairVarTypeName)),
                      hasInitializer(expr()))
                  .bind(PairDeclName)),
          hasNextTwoVarDecl(VarInitWithFirstMember, VarInitWithSecondMember)),
      this);

  // for (pair<X, Y> p : map) {
  //    X x = p.first;
  //    Y y = p.second;
  // }
  Finder->addMatcher(
      cxxForRangeStmt(
          unless(isInMarco()),
          hasLoopVariable(
              varDecl(hasType(qualType(anyOf(PairType, lValueReferenceType(
                                                           pointee(PairType))))
                                  .bind(PairVarTypeName)),
                      hasInitializer(expr()))
                  .bind(PairDeclName)),
          hasBody(compoundStmt(hasFirstTwoVarDecl(VarInitWithFirstMember,
                                                  VarInitWithSecondMember))
                      .bind(ScopeBlockName)))
          .bind(ForRangeStmtName),
      this);
}

static std::optional<TransferType> getTransferType(const ASTContext &Ctx,
                                                   QualType ResultType,
                                                   QualType OriginType) {
  ResultType = ResultType.getCanonicalType();
  OriginType = OriginType.getCanonicalType();

  if (ResultType == Ctx.getLValueReferenceType(OriginType.withConst()))
    return TT_ByConstRef;

  if (ResultType == Ctx.getLValueReferenceType(OriginType))
    return TT_ByRef;

  if (ResultType == OriginType.withConst())
    return TT_ByConstVal;

  if (ResultType == OriginType)
    return TT_ByVal;

  return std::nullopt;
}

void UseStructuredBindingCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *FirstVar = Result.Nodes.getNodeAs<VarDecl>(FirstVarDeclName);
  const auto *SecondVar = Result.Nodes.getNodeAs<VarDecl>(SecondVarDeclName);

  const auto *DS1 = Result.Nodes.getNodeAs<DeclStmt>(FirstDeclStmtName);
  const auto *DS2 = Result.Nodes.getNodeAs<DeclStmt>(SecondDeclStmtName);
  const auto *ScopeBlock = Result.Nodes.getNodeAs<CompoundStmt>(ScopeBlockName);

  // Captured structured bindings are a C++20 extension
  if (!Result.Context->getLangOpts().CPlusPlus20) {
    if (auto Matchers = match(
            compoundStmt(
                hasDescendant(lambdaExpr(hasAnyCapture(capturesVar(varDecl(
                    anyOf(equalsNode(FirstVar), equalsNode(SecondVar)))))))),
            *ScopeBlock, *Result.Context);
        !Matchers.empty())
      return;
  }

  const auto *CFRS = Result.Nodes.getNodeAs<CXXForRangeStmt>(ForRangeStmtName);
  auto DiagAndFix = [&](SourceLocation DiagLoc, SourceRange ReplaceRange,
                        TransferType TT = TT_ByVal) {
    StringRef Prefix;
    switch (TT) {
    case TT_ByVal:
      Prefix = "auto";
      break;
    case TT_ByConstVal:
      Prefix = "const auto";
      break;
    case TT_ByRef:
      Prefix = "auto&";
      break;
    case TT_ByConstRef:
      Prefix = "const auto&";
      break;
    }
    std::vector<FixItHint> Hints;
    if (DS1)
      Hints.emplace_back(FixItHint::CreateRemoval(DS1->getSourceRange()));
    if (DS2)
      Hints.emplace_back(FixItHint::CreateRemoval(DS2->getSourceRange()));

    std::string ReplacementText = Prefix.str() + " [" +
                                  FirstVar->getNameAsString() + ", " +
                                  SecondVar->getNameAsString() + "]";
    if (CFRS)
      ReplacementText += " :";
    diag(DiagLoc, "use structured binding to decompose a pair")
        << FixItHint::CreateReplacement(ReplaceRange, ReplacementText) << Hints;
  };

  if (const auto *COCE =
          Result.Nodes.getNodeAs<CXXOperatorCallExpr>(StdTieAssignStmtName)) {
    DiagAndFix(COCE->getBeginLoc(),
               Result.Nodes.getNodeAs<Expr>(StdTieExprName)->getSourceRange());
    return;
  }

  // Check whether PairVar, FirstVar and SecondVar have the same transfer type,
  // so they can be combined to structured binding.
  const auto *PairVar = Result.Nodes.getNodeAs<VarDecl>(PairDeclName);
  const Expr *InitE = PairVar->getInit();
  if (auto Res =
          match(expr(ignoringCopyCtorAndImplicitCast(expr().bind("init_expr"))),
                *InitE, *Result.Context);
      !Res.empty())
    InitE = Res[0].getNodeAs<Expr>("init_expr");

  std::optional<TransferType> PairCaptureType =
      getTransferType(*Result.Context, PairVar->getType(), InitE->getType());
  std::optional<TransferType> FirstVarCaptureType =
      getTransferType(*Result.Context, FirstVar->getType(),
                      *Result.Nodes.getNodeAs<QualType>(FirstTypeName));
  std::optional<TransferType> SecondVarCaptureType =
      getTransferType(*Result.Context, SecondVar->getType(),
                      *Result.Nodes.getNodeAs<QualType>(SecondTypeName));
  if (!PairCaptureType || !FirstVarCaptureType || !SecondVarCaptureType ||
      *PairCaptureType != *FirstVarCaptureType ||
      *FirstVarCaptureType != *SecondVarCaptureType)
    return;

  // Check PairVar is not used except for assignment members to firstVar and
  // SecondVar.
  if (auto AllRef = utils::decl_ref_expr::allDeclRefExprs(*PairVar, *ScopeBlock,
                                                          *Result.Context);
      AllRef.size() != 2)
    return;

  DiagAndFix(PairVar->getBeginLoc(),
             CFRS ? PairVar->getSourceRange()
                  : SourceRange(PairVar->getBeginLoc(),
                                Lexer::getLocForEndOfToken(
                                    PairVar->getLocation(), 0,
                                    Result.Context->getSourceManager(),
                                    Result.Context->getLangOpts())),
             *PairCaptureType);
}

} // namespace clang::tidy::modernize
