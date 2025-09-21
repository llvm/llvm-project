//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseStructuredBindingCheck.h"
#include "../utils/DeclRefExprUtils.h"
#include "clang/Lex/Lexer.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static constexpr llvm::StringLiteral PairDeclName = "PairVarD";
static constexpr llvm::StringLiteral PairVarTypeName = "PairVarType";
static constexpr llvm::StringLiteral FirstVarDeclName = "FirstVarDecl";
static constexpr llvm::StringLiteral SecondVarDeclName = "SecondVarDecl";
static constexpr llvm::StringLiteral BeginDeclStmtName = "BeginDeclStmt";
static constexpr llvm::StringLiteral EndDeclStmtName = "EndDeclStmt";
static constexpr llvm::StringLiteral FirstTypeName = "FirstType";
static constexpr llvm::StringLiteral SecondTypeName = "SecondType";
static constexpr llvm::StringLiteral ScopeBlockName = "ScopeBlock";
static constexpr llvm::StringLiteral StdTieAssignStmtName = "StdTieAssign";
static constexpr llvm::StringLiteral StdTieExprName = "StdTieExpr";
static constexpr llvm::StringLiteral ForRangeStmtName = "ForRangeStmt";
static constexpr llvm::StringLiteral InitExprName = "init_expr";

/// Matches a sequence of VarDecls matching the inner matchers, starting from
/// the \p Iter to \p EndIter and set bindings for the first DeclStmt and the
/// last DeclStmt if matched.
///
/// \p Backwards indicates whether to match the VarDecls in reverse order.
template <typename Iterator>
static bool matchNVarDeclStartingWith(
    Iterator Iter, Iterator EndIter,
    ArrayRef<ast_matchers::internal::Matcher<VarDecl>> InnerMatchers,
    ast_matchers::internal::ASTMatchFinder *Finder,
    ast_matchers::internal::BoundNodesTreeBuilder *Builder,
    bool Backwards = false) {
  const DeclStmt *BeginDS = nullptr;
  const DeclStmt *EndDS = nullptr;
  size_t N = InnerMatchers.size();
  size_t Count = 0;
  for (; Iter != EndIter; ++Iter) {
    EndDS = dyn_cast<DeclStmt>(*Iter);
    if (!EndDS)
      break;

    if (!BeginDS)
      BeginDS = EndDS;

    auto Matches = [&](const Decl *VD) {
      // We don't want redundant decls in DeclStmt.
      if (Count == N)
        return false;

      if (const auto *Var = dyn_cast<VarDecl>(VD);
          Var && InnerMatchers[Backwards ? N - Count - 1 : Count].matches(
                     *Var, Finder, Builder)) {
        ++Count;
        return true;
      }

      return false;
    };

    if (Backwards) {
      for (const auto *VD : llvm::reverse(EndDS->decls())) {
        if (!Matches(VD))
          return false;
      }
    } else {
      for (const auto *VD : EndDS->decls()) {
        if (!Matches(VD))
          return false;
      }
    }

    // All the matchers is satisfied in those DeclStmts.
    if (Count == N) {
      Builder->setBinding(
          BeginDeclStmtName,
          clang::DynTypedNode::create(Backwards ? *EndDS : *BeginDS));
      Builder->setBinding(EndDeclStmtName, clang::DynTypedNode::create(
                                               Backwards ? *BeginDS : *EndDS));
      return true;
    }
  }

  return false;
}

namespace {
/// What qualifiers and specifiers are used to create structured binding
/// declaration, it only supports the following four cases now.
enum TransferType : uint8_t {
  TT_ByVal,
  TT_ByConstVal,
  TT_ByRef,
  TT_ByConstRef
};

/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// following two VarDecls matching the inner matcher.
AST_MATCHER_P(Stmt, hasPreTwoVarDecl,
              llvm::SmallVector<ast_matchers::internal::Matcher<VarDecl>>,
              InnerMatchers) {
  const DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  const auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto I = llvm::find(llvm::reverse(C->body()), &Node);
  assert(I != C->body_rend() && "C is parent of Node");
  return matchNVarDeclStartingWith(I + 1, C->body_rend(), InnerMatchers, Finder,
                                   Builder, true);
}

/// Matches a Stmt whose parent is a CompoundStmt, and which is directly
/// followed by two VarDecls matching the inner matcher.
AST_MATCHER_P(Stmt, hasNextTwoVarDecl,
              llvm::SmallVector<ast_matchers::internal::Matcher<VarDecl>>,
              InnerMatchers) {
  const DynTypedNodeList Parents = Finder->getASTContext().getParents(Node);
  if (Parents.size() != 1)
    return false;

  const auto *C = Parents[0].get<CompoundStmt>();
  if (!C)
    return false;

  const auto *I = llvm::find(C->body(), &Node);
  assert(I != C->body_end() && "C is parent of Node");
  return matchNVarDeclStartingWith(I + 1, C->body_end(), InnerMatchers, Finder,
                                   Builder);
}

/// Matches a CompoundStmt which has two VarDecls matching the inner matcher in
/// the beginning.
AST_MATCHER_P(CompoundStmt, hasFirstTwoVarDecl,
              llvm::SmallVector<ast_matchers::internal::Matcher<VarDecl>>,
              InnerMatchers) {
  return matchNVarDeclStartingWith(Node.body_begin(), Node.body_end(),
                                   InnerMatchers, Finder, Builder);
}

/// It's not very common to have specifiers for variables used to decompose a
/// pair, so we ignore these cases.
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
    if (const CXXConstructorDecl *CtorD = CtorE->getConstructor();
        CtorD->isCopyConstructor() && CtorE->getNumArgs() == 1) {
      return InnerMatcher.matches(*CtorE->getArg(0)->IgnoreImpCasts(), Finder,
                                  Builder);
    }
  }

  return InnerMatcher.matches(*Node.IgnoreImpCasts(), Finder, Builder);
}

AST_MATCHER(CXXRecordDecl, isPairType) {
  return llvm::all_of(Node.fields(), [](const FieldDecl *FD) {
    return FD->getAccess() == AS_public &&
           (FD->getName() == "first" || FD->getName() == "second");
  });
}

} // namespace

static auto getVarInitWithMemberMatcher(
    StringRef PairName, StringRef MemberName, StringRef TypeName,
    StringRef BindingName,
    ast_matchers::internal::Matcher<VarDecl> ExtraMatcher) {
  return varDecl(
             ExtraMatcher,
             hasInitializer(
                 ignoringImpCasts(ignoringCopyCtorAndImplicitCast(memberExpr(
                     hasObjectExpression(ignoringImpCasts(declRefExpr(
                         to(equalsBoundNode(std::string(PairName)))))),
                     member(fieldDecl(hasName(MemberName),
                                      hasType(qualType().bind(TypeName)))))))))
      .bind(BindingName);
}

void UseStructuredBindingCheck::registerMatchers(MatchFinder *Finder) {
  auto PairType = qualType(unless(isVolatileQualified()),
                           hasUnqualifiedDesugaredType(recordType(
                               hasDeclaration(cxxRecordDecl(isPairType())))));

  auto UnlessShouldBeIgnored =
      unless(anyOf(hasAnySpecifiersShouldBeIgnored(), isInMarco()));

  auto VarInitWithFirstMember =
      getVarInitWithMemberMatcher(PairDeclName, "first", FirstTypeName,
                                  FirstVarDeclName, UnlessShouldBeIgnored);
  auto VarInitWithSecondMember =
      getVarInitWithMemberMatcher(PairDeclName, "second", SecondTypeName,
                                  SecondVarDeclName, UnlessShouldBeIgnored);

  auto RefToBindName = [&UnlessShouldBeIgnored](const llvm::StringLiteral &Name)
      -> ast_matchers::internal::BindableMatcher<Stmt> {
    return declRefExpr(to(varDecl(UnlessShouldBeIgnored).bind(Name)));
  };

  auto HasAnyLambdaCaptureThisVar =
      [](ast_matchers::internal::Matcher<VarDecl> VDMatcher)
      -> ast_matchers::internal::BindableMatcher<Stmt> {
    return compoundStmt(hasDescendant(
        lambdaExpr(hasAnyCapture(capturesVar(varDecl(VDMatcher))))));
  };

  // Captured structured bindings are a C++20 extension
  auto UnlessFirstVarOrSecondVarIsCapturedByLambda =
      getLangOpts().CPlusPlus20
          ? compoundStmt()
          : compoundStmt(unless(HasAnyLambdaCaptureThisVar(
                anyOf(equalsBoundNode(std::string(FirstVarDeclName)),
                      equalsBoundNode(std::string(SecondVarDeclName))))));

  // X x;
  // Y y;
  // std::tie(x, y) = ...;
  Finder->addMatcher(
      exprWithCleanups(
          unless(isInMarco()),
          has(cxxOperatorCallExpr(
                  hasOverloadedOperatorName("="),
                  hasLHS(ignoringImplicit(
                      callExpr(callee(functionDecl(isInStdNamespace(),
                                                   hasName("tie"))),
                               hasArgument(0, RefToBindName(FirstVarDeclName)),
                               hasArgument(1, RefToBindName(SecondVarDeclName)))
                          .bind(StdTieExprName))),
                  hasRHS(expr(hasType(PairType))))
                  .bind(StdTieAssignStmtName)),
          hasPreTwoVarDecl(
              llvm::SmallVector<ast_matchers::internal::Matcher<VarDecl>>{
                  varDecl(equalsBoundNode(std::string(FirstVarDeclName))),
                  varDecl(equalsBoundNode(std::string(SecondVarDeclName)))}),
          hasParent(compoundStmt(UnlessFirstVarOrSecondVarIsCapturedByLambda)
                        .bind(ScopeBlockName))),
      this);

  // pair<X, Y> p = ...;
  // X x = p.first;
  // Y y = p.second;
  Finder->addMatcher(
      declStmt(
          unless(isInMarco()),
          hasSingleDecl(
              varDecl(UnlessShouldBeIgnored,
                      hasType(qualType(anyOf(PairType, lValueReferenceType(
                                                           pointee(PairType))))
                                  .bind(PairVarTypeName)),
                      hasInitializer(expr(ignoringCopyCtorAndImplicitCast(
                          expr().bind(InitExprName)))))
                  .bind(PairDeclName)),
          hasNextTwoVarDecl(
              llvm::SmallVector<ast_matchers::internal::Matcher<VarDecl>>{
                  VarInitWithFirstMember, VarInitWithSecondMember}),
          hasParent(compoundStmt(UnlessFirstVarOrSecondVarIsCapturedByLambda)
                        .bind(ScopeBlockName))),
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
                      hasInitializer(expr(ignoringCopyCtorAndImplicitCast(
                          expr().bind(InitExprName)))))
                  .bind(PairDeclName)),
          hasBody(
              compoundStmt(
                  hasFirstTwoVarDecl(llvm::SmallVector<
                                     ast_matchers::internal::Matcher<VarDecl>>{
                      VarInitWithFirstMember, VarInitWithSecondMember}),
                  UnlessFirstVarOrSecondVarIsCapturedByLambda)
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

  const auto *BeginDS = Result.Nodes.getNodeAs<DeclStmt>(BeginDeclStmtName);
  const auto *EndDS = Result.Nodes.getNodeAs<DeclStmt>(EndDeclStmtName);
  const auto *ScopeBlock = Result.Nodes.getNodeAs<CompoundStmt>(ScopeBlockName);

  const auto *CFRS = Result.Nodes.getNodeAs<CXXForRangeStmt>(ForRangeStmtName);
  auto DiagAndFix = [&BeginDS, &EndDS, &FirstVar, &SecondVar, &CFRS,
                     this](SourceLocation DiagLoc, SourceRange ReplaceRange,
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

    std::string ReplacementText =
        (Twine(Prefix) + " [" + FirstVar->getNameAsString() + ", " +
         SecondVar->getNameAsString() + "]" + (CFRS ? " :" : ""))
            .str();
    diag(DiagLoc, "use a structured binding to decompose a pair")
        << FixItHint::CreateReplacement(ReplaceRange, ReplacementText)
        << FixItHint::CreateRemoval(
               SourceRange{BeginDS->getBeginLoc(), EndDS->getEndLoc()});
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

  const std::optional<TransferType> PairCaptureType =
      getTransferType(*Result.Context, PairVar->getType(),
                      Result.Nodes.getNodeAs<Expr>(InitExprName)->getType());
  const std::optional<TransferType> FirstVarCaptureType =
      getTransferType(*Result.Context, FirstVar->getType(),
                      *Result.Nodes.getNodeAs<QualType>(FirstTypeName));
  const std::optional<TransferType> SecondVarCaptureType =
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
