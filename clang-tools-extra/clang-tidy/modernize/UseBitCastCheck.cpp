//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBitCastCheck.h"
#include "../utils/Matchers.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static const Expr *stripMemcpyArgument(const Expr *ExprNode) {
  ExprNode = ExprNode->IgnoreParenImpCasts();
  while (const auto *Cast = dyn_cast<ExplicitCastExpr>(ExprNode))
    ExprNode = Cast->getSubExpr()->IgnoreParenImpCasts();
  return ExprNode;
}

static bool isSupportedMemcpyObjectExpr(const Expr *ExprNode) {
  ExprNode = ExprNode->IgnoreParenImpCasts();

  if (isa<DeclRefExpr>(ExprNode))
    return true;

  const auto *Member = dyn_cast<MemberExpr>(ExprNode);
  if (!Member || !isa<FieldDecl>(Member->getMemberDecl()))
    if (const auto *MemberPointer = dyn_cast<BinaryOperator>(ExprNode))
      if (MemberPointer->getOpcode() == BO_PtrMemD ||
          MemberPointer->getOpcode() == BO_PtrMemI)
        return isSupportedMemcpyObjectExpr(MemberPointer->getLHS());

  return Member && isSupportedMemcpyObjectExpr(Member->getBase());
}

static const Expr *extractMemcpyObjectExpr(const Expr *ExprNode) {
  ExprNode = stripMemcpyArgument(ExprNode);
  const auto *AddressOf = dyn_cast<UnaryOperator>(ExprNode);
  if (!AddressOf || AddressOf->getOpcode() != UO_AddrOf)
    return nullptr;

  const Expr *ObjectExpr = AddressOf->getSubExpr()->IgnoreParenImpCasts();
  return isSupportedMemcpyObjectExpr(ObjectExpr) ? ObjectExpr : nullptr;
}

static bool isSupportedMemcpyArgType(QualType Type, const ASTContext &Context,
                                     bool RequireMutable) {
  if (Type.isNull())
    return false;

  const QualType CanonicalType = Type.getCanonicalType().getNonReferenceType();
  if (CanonicalType.isNull() || CanonicalType->isDependentType() ||
      CanonicalType->isIncompleteType() ||
      CanonicalType.isVolatileQualified() ||
      CanonicalType->isAnyPointerType() || CanonicalType->isArrayType() ||
      CanonicalType->isFunctionType())
    return false;

  if (RequireMutable) {
    if (CanonicalType.isConstQualified())
      return false;

    if (const auto *Record = CanonicalType->getAsCXXRecordDecl())
      if (!Record->hasSimpleCopyAssignment() &&
          !Record->hasSimpleMoveAssignment())
        return false;
  }

  return Type.getNonReferenceType().isTriviallyCopyableType(Context);
}

static bool isSameUnqualifiedCanonicalType(QualType LHS, QualType RHS) {
  return LHS.getCanonicalType().getUnqualifiedType() ==
         RHS.getCanonicalType().getUnqualifiedType();
}

static bool isMatchingSizeOfExpression(const Expr *SizeExpr, QualType SrcType,
                                       QualType DstType,
                                       const ASTContext &Context) {
  const auto *UnaryExpr =
      dyn_cast<UnaryExprOrTypeTraitExpr>(SizeExpr->IgnoreParenImpCasts());
  if (!UnaryExpr || UnaryExpr->getKind() != UETT_SizeOf ||
      SizeExpr->getBeginLoc().isMacroID())
    return false;

  const QualType SizeType = UnaryExpr->getTypeOfArgument();
  if (SizeType.isNull())
    return false;

  const QualType SizeCanonical =
      SizeType.getCanonicalType().getUnqualifiedType();
  const QualType SrcCanonical = SrcType.getCanonicalType().getUnqualifiedType();
  const QualType DstCanonical = DstType.getCanonicalType().getUnqualifiedType();
  if (SizeCanonical != SrcCanonical && SizeCanonical != DstCanonical)
    return false;

  return Context.getTypeSizeInChars(SrcCanonical) ==
         Context.getTypeSizeInChars(DstCanonical);
}

static bool isStatementBody(const Stmt *Current, const Stmt *Parent) {
  if (const auto *Block = dyn_cast<CompoundStmt>(Parent))
    return llvm::is_contained(Block->body(), Current);

  if (const auto *If = dyn_cast<IfStmt>(Parent))
    return If->getThen() == Current || If->getElse() == Current;
  if (const auto *While = dyn_cast<WhileStmt>(Parent))
    return While->getBody() == Current;
  if (const auto *Do = dyn_cast<DoStmt>(Parent))
    return Do->getBody() == Current;
  if (const auto *For = dyn_cast<ForStmt>(Parent))
    return For->getBody() == Current;
  if (const auto *RangeFor = dyn_cast<CXXForRangeStmt>(Parent))
    return RangeFor->getBody() == Current;
  if (const auto *Label = dyn_cast<LabelStmt>(Parent))
    return Label->getSubStmt() == Current;
  if (const auto *Case = dyn_cast<SwitchCase>(Parent))
    return Case->getSubStmt() == Current;
  if (const auto *Attributed = dyn_cast<AttributedStmt>(Parent))
    return Attributed->getSubStmt() == Current;

  return false;
}

namespace {

// These states describe how to spell the replacement when only the memcpy call
// is replaced. An existing `(void)` cast is preserved by parenthesizing the
// assignment, while comma/discarded subexpressions need an injected `(void)`.
enum class MemcpyReplacementForm {
  None,
  StatementBody,
  PreserveOuterVoidCast,
  InjectVoidCast,
};

} // namespace

static MemcpyReplacementForm getMemcpyReplacementForm(const Expr *ExprNode,
                                                      ASTContext &Context) {
  const Stmt *Current = ExprNode;
  MemcpyReplacementForm Kind = MemcpyReplacementForm::StatementBody;

  while (true) {
    auto Parents = Context.getParents(*Current);
    if (Parents.size() != 1)
      return MemcpyReplacementForm::None;

    if (const auto *ParentExpr = Parents[0].get<Expr>()) {
      if (isa<ExprWithCleanups, ImplicitCastExpr, MaterializeTemporaryExpr,
              CXXBindTemporaryExpr, ParenExpr>(ParentExpr)) {
        Current = ParentExpr;
        continue;
      }

      if (const auto *Cast = dyn_cast<CastExpr>(ParentExpr))
        if (Cast->getCastKind() == CK_ToVoid)
          return MemcpyReplacementForm::PreserveOuterVoidCast;

      if (const auto *Comma = dyn_cast<BinaryOperator>(ParentExpr)) {
        if (Comma->getOpcode() != BO_Comma)
          return MemcpyReplacementForm::None;
        if (Comma->getLHS() == Current)
          return MemcpyReplacementForm::InjectVoidCast;
        if (Comma->getRHS() == Current) {
          Current = Comma;
          Kind = MemcpyReplacementForm::InjectVoidCast;
          continue;
        }
      }

      return MemcpyReplacementForm::None;
    }

    const auto *ParentStmt = Parents[0].get<Stmt>();
    if (!ParentStmt || !isStatementBody(Current, ParentStmt))
      return MemcpyReplacementForm::None;
    return Kind;
  }
}

namespace {

AST_MATCHER(CallExpr, isDiscardedValueContext) {
  return getMemcpyReplacementForm(&Node, Finder->getASTContext()) !=
         MemcpyReplacementForm::None;
}

AST_MATCHER(CallExpr, isBitCastMemcpyCandidate) {
  if (Node.getNumArgs() != 3 || Node.getBeginLoc().isMacroID())
    return false;

  const auto *DstExpr = extractMemcpyObjectExpr(Node.getArg(0));
  const auto *SrcExpr = extractMemcpyObjectExpr(Node.getArg(1));
  if (!DstExpr || !SrcExpr || DstExpr->getBeginLoc().isMacroID() ||
      SrcExpr->getBeginLoc().isMacroID())
    return false;

  const auto &Context = Finder->getASTContext();
  const QualType DstType = DstExpr->getType().getNonReferenceType();
  const QualType SrcType = SrcExpr->getType().getNonReferenceType();

  return isSupportedMemcpyArgType(DstType, Context, /*RequireMutable=*/true) &&
         isSupportedMemcpyArgType(SrcType, Context,
                                  /*RequireMutable=*/false) &&
         !isSameUnqualifiedCanonicalType(SrcType, DstType) &&
         isMatchingSizeOfExpression(Node.getArg(2), SrcType, DstType, Context);
}

} // namespace

static StringRef getSourceText(const Expr *ExprNode, const SourceManager &SM,
                               const LangOptions &LangOpts) {
  return Lexer::getSourceText(
      CharSourceRange::getTokenRange(ExprNode->getSourceRange()), SM, LangOpts);
}

UseBitCastCheck::UseBitCastCheck(StringRef Name, ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context),
      IncludeInserter(Options.getLocalOrGlobal("IncludeStyle",
                                               utils::IncludeSorter::IS_LLVM),
                      areDiagsSelfContained()) {}

void UseBitCastCheck::storeOptions(ClangTidyOptions::OptionMap &Opts) {
  Options.store(Opts, "IncludeStyle", IncludeInserter.getStyle());
}

void UseBitCastCheck::registerPPCallbacks(const SourceManager &SM,
                                          Preprocessor *PP,
                                          Preprocessor *ModuleExpanderPP) {
  IncludeInserter.registerPreprocessor(PP);
}

void UseBitCastCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      callExpr(callee(functionDecl(hasName("::memcpy"))),
               isDiscardedValueContext(), unless(isInTemplateInstantiation()),
               unless(hasAncestor(expr(matchers::hasUnevaluatedContext()))),
               isBitCastMemcpyCandidate())
          .bind("memcpy"),
      this);
}

void UseBitCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemcpyCall = Result.Nodes.getNodeAs<CallExpr>("memcpy");
  if (!MemcpyCall)
    return;

  const auto *DstExpr = extractMemcpyObjectExpr(MemcpyCall->getArg(0));
  const auto *SrcExpr = extractMemcpyObjectExpr(MemcpyCall->getArg(1));
  if (!DstExpr || !SrcExpr)
    return;

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();
  StringRef DstText = getSourceText(DstExpr, SM, LangOpts);
  StringRef SrcText = getSourceText(SrcExpr, SM, LangOpts);
  if (DstText.empty() || SrcText.empty())
    return;

  const MemcpyReplacementForm ReplacementForm =
      getMemcpyReplacementForm(MemcpyCall, *Result.Context);
  if (ReplacementForm == MemcpyReplacementForm::None)
    return;

  const PrintingPolicy Policy(LangOpts);
  const QualType DstType =
      DstExpr->getType().getNonReferenceType().getUnqualifiedType();
  const std::string Assignment = std::string(DstText) + " = std::bit_cast<" +
                                 DstType.getAsString(Policy) + ">(" +
                                 std::string(SrcText) + ")";
  std::string Replacement = Assignment;
  switch (ReplacementForm) {
  case MemcpyReplacementForm::StatementBody:
    break;
  case MemcpyReplacementForm::PreserveOuterVoidCast:
    Replacement = "(" + Assignment + ")";
    break;
  case MemcpyReplacementForm::InjectVoidCast:
    Replacement = "(void)(" + Assignment + ")";
    break;
  case MemcpyReplacementForm::None:
    return;
  }

  const DiagnosticBuilder Diag =
      diag(MemcpyCall->getBeginLoc(),
           "use 'std::bit_cast' instead of 'memcpy' for type punning");
  Diag << FixItHint::CreateReplacement(MemcpyCall->getSourceRange(),
                                       Replacement);
  Diag << IncludeInserter.createIncludeInsertion(
      SM.getFileID(MemcpyCall->getBeginLoc()), "<bit>");
}

} // namespace clang::tidy::modernize
