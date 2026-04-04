//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBitCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Twine.h"
#include <cassert>

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

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
  ExprNode = ExprNode->IgnoreParenCasts();
  const auto *AddressOf = dyn_cast<UnaryOperator>(ExprNode);
  if (!AddressOf || AddressOf->getOpcode() != UO_AddrOf)
    return nullptr;

  const Expr *ObjectExpr = AddressOf->getSubExpr()->IgnoreParenImpCasts();
  return isSupportedMemcpyObjectExpr(ObjectExpr) ? ObjectExpr : nullptr;
}

static bool isBitCastableMemcpyObjectType(QualType Type,
                                          const ASTContext &Context) {
  Type = Type.getCanonicalType().getNonReferenceType();
  return !Type.isNull() && !Type.isVolatileQualified() &&
         !Type->isAnyPointerType() && !Type->isFunctionType() &&
         Type.isTriviallyCopyableType(Context) &&
         Type.isBitwiseCloneableType(Context);
}

static bool canAssignBitCastResult(QualType Type) {
  Type = Type.getCanonicalType().getNonReferenceType();
  if (Type.isNull() || Type.isConstQualified() || Type->isArrayType())
    return false;

  const auto *Record = Type->getAsCXXRecordDecl();
  return !Record || Record->hasSimpleCopyAssignment() ||
         Record->hasSimpleMoveAssignment();
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

AST_MATCHER(CallExpr, hasBitCastReplacementContext) {
  const Stmt *Current = &Node;
  const BinaryOperator *DiscardedComma = nullptr;

  while (true) {
    auto Parents = Finder->getASTContext().getParents(*Current);
    if (Parents.size() != 1)
      return false;

    if (const auto *ParentExpr = Parents[0].get<Expr>()) {
      if (isa<ExprWithCleanups, ImplicitCastExpr, MaterializeTemporaryExpr,
              CXXBindTemporaryExpr, ParenExpr>(ParentExpr)) {
        Current = ParentExpr;
        continue;
      }

      if (const auto *Cast = dyn_cast<CastExpr>(ParentExpr)) {
        if (Cast->getCastKind() != CK_ToVoid)
          return false;

        if (!DiscardedComma) {
          Builder->setBinding("replacementRoot", DynTypedNode::create(*Cast));
          Builder->setBinding("discardedVoidCast", DynTypedNode::create(*Cast));
          return true;
        }

        Current = Cast;
        continue;
      }

      const auto *Comma = dyn_cast<BinaryOperator>(ParentExpr);
      if (!Comma || Comma->getOpcode() != BO_Comma)
        return false;
      if (Comma->getLHS() == Current) {
        Builder->setBinding("replacementRoot", DynTypedNode::create(Node));
        Builder->setBinding("discardedComma", DynTypedNode::create(*Comma));
        return true;
      }
      if (Comma->getRHS() != Current)
        return false;

      DiscardedComma = Comma;
      Current = Comma;
      continue;
    }

    const auto *ParentStmt = Parents[0].get<Stmt>();
    if (!ParentStmt || !isStatementBody(Current, ParentStmt))
      return false;

    Builder->setBinding("replacementRoot", DynTypedNode::create(Node));
    if (DiscardedComma)
      Builder->setBinding("discardedComma",
                          DynTypedNode::create(*DiscardedComma));
    return true;
  }
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

  if (!isBitCastableMemcpyObjectType(DstType, Context) ||
      !isBitCastableMemcpyObjectType(SrcType, Context) ||
      !canAssignBitCastResult(DstType) ||
      isSameUnqualifiedCanonicalType(SrcType, DstType) ||
      !isMatchingSizeOfExpression(Node.getArg(2), SrcType, DstType, Context))
    return false;

  Builder->setBinding("dstExpr", DynTypedNode::create(*DstExpr));
  Builder->setBinding("srcExpr", DynTypedNode::create(*SrcExpr));
  return true;
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
  Finder->addMatcher(callExpr(callee(functionDecl(hasName("::memcpy"))),
                              hasBitCastReplacementContext(),
                              isBitCastMemcpyCandidate())
                         .bind("memcpy"),
                     this);
}

void UseBitCastCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MemcpyCall = Result.Nodes.getNodeAs<CallExpr>("memcpy");
  const auto *DstExpr = Result.Nodes.getNodeAs<Expr>("dstExpr");
  const auto *SrcExpr = Result.Nodes.getNodeAs<Expr>("srcExpr");
  const auto *ReplacementRoot = Result.Nodes.getNodeAs<Expr>("replacementRoot");
  const auto *DiscardedComma =
      Result.Nodes.getNodeAs<BinaryOperator>("discardedComma");
  const auto *DiscardedVoidCast =
      Result.Nodes.getNodeAs<CastExpr>("discardedVoidCast");
  assert(MemcpyCall);
  assert(DstExpr);
  assert(SrcExpr);
  assert(ReplacementRoot);
  assert(!DiscardedVoidCast || ReplacementRoot == DiscardedVoidCast);

  const SourceManager &SM = *Result.SourceManager;
  const LangOptions &LangOpts = getLangOpts();
  StringRef DstText = getSourceText(DstExpr, SM, LangOpts);
  StringRef SrcText = getSourceText(SrcExpr, SM, LangOpts);
  if (DstText.empty() || SrcText.empty())
    return;

  const PrintingPolicy &Policy = Result.Context->getPrintingPolicy();
  const QualType DstType =
      DstExpr->getType().getNonReferenceType().getUnqualifiedType();
  const std::string DstTypeName = DstType.getAsString(Policy);
  const std::string Replacement =
      [&](const llvm::Twine &Assignment) -> std::string {
    if (DiscardedComma)
      return ("(void)(" + Assignment + ")").str();
    return Assignment.str();
  }(llvm::Twine(DstText) + " = std::bit_cast<" + DstTypeName + ">(" + SrcText +
                                         ")");

  const DiagnosticBuilder Diag =
      diag(MemcpyCall->getBeginLoc(),
           "use 'std::bit_cast' instead of 'memcpy' for type punning");
  Diag << FixItHint::CreateReplacement(ReplacementRoot->getSourceRange(),
                                       Replacement);
  Diag << IncludeInserter.createIncludeInsertion(
      SM.getFileID(MemcpyCall->getBeginLoc()), "<bit>");
}

} // namespace clang::tidy::modernize
