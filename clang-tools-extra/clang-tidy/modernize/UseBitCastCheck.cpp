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
#include "llvm/ADT/TypeSwitch.h"

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
  const auto IsCurrentBody = [Current](const Stmt *Body) {
    if (Body == Current)
      return true;

    // IgnoreUnlessSpelledInSource can make `Current` skip over a parenthesized
    // body expression even though the enclosing statement still stores it.
    const auto *BodyExpr = dyn_cast_or_null<Expr>(Body);
    return BodyExpr && BodyExpr->IgnoreParenImpCasts() == Current;
  };

  return llvm::TypeSwitch<const Stmt *, bool>(Parent)
      .Case<CompoundStmt>([&](const CompoundStmt *Block) {
        return llvm::any_of(Block->body(), IsCurrentBody);
      })
      .Case<IfStmt>([&](const IfStmt *If) {
        return IsCurrentBody(If->getThen()) || IsCurrentBody(If->getElse());
      })
      .Case<WhileStmt, DoStmt, ForStmt, CXXForRangeStmt>(
          [&](const auto *Loop) { return IsCurrentBody(Loop->getBody()); })
      .Case<LabelStmt, SwitchCase, AttributedStmt>([&](const auto *Wrapper) {
        return IsCurrentBody(Wrapper->getSubStmt());
      })
      .Default(false);
}

namespace {

// Accept only discarded-value uses of the memcpy call:
//   memcpy(...);
//   (void)memcpy(...);
//   (memcpy(...), rhs);
//   (lhs, memcpy(...));    if the enclosing comma expression is discarded
//   (void)(lhs, memcpy(...));
// Skip transparent wrappers on the way up and reject any other parent shape.
AST_MATCHER(CallExpr, hasBitCastReplacementContext) {
  const Stmt *Current = &Node;
  bool SawDiscardedCommaRHS = false;
  const auto IsTransparentReplacementParent = [](const Expr *ExprNode) {
    return isa<ExprWithCleanups, ImplicitCastExpr, MaterializeTemporaryExpr,
               CXXBindTemporaryExpr, ParenExpr>(ExprNode);
  };
  const auto BindReplacementContext = [&](const Expr &ReplacementRoot,
                                          const BinaryOperator *CommaLHS) {
    Builder->setBinding("replacementRoot",
                        DynTypedNode::create(ReplacementRoot));
    if (CommaLHS)
      Builder->setBinding("commaLHS", DynTypedNode::create(*CommaLHS));
    return true;
  };

  while (true) {
    auto Parents = Finder->getASTContext().getParents(*Current);
    if (Parents.size() != 1)
      return false;

    if (const auto *ParentExpr = Parents[0].get<Expr>()) {
      if (IsTransparentReplacementParent(ParentExpr)) {
        Current = ParentExpr;
        continue;
      }

      if (const auto *Cast = dyn_cast<CastExpr>(ParentExpr)) {
        if (Cast->getCastKind() != CK_ToVoid)
          return false;
        if (!SawDiscardedCommaRHS)
          return BindReplacementContext(*Cast, nullptr);

        Current = Cast;
        continue;
      }

      const auto *Comma = dyn_cast<BinaryOperator>(ParentExpr);
      if (!Comma || Comma->getOpcode() != BO_Comma)
        return false;
      if (Comma->getLHS() == Current)
        return BindReplacementContext(Node, Comma);
      if (Comma->getRHS() != Current)
        return false;

      // A memcpy on the right-hand side of `,` is safe only if the enclosing
      // comma expression is itself discarded, so keep walking from the comma
      // node. Inject `(void)` only if that comma expression later becomes the
      // left-hand side of another comma.
      SawDiscardedCommaRHS = true;
      Current = Comma;
      continue;
    }

    const auto *ParentStmt = Parents[0].get<Stmt>();
    if (!ParentStmt || !isStatementBody(Current, ParentStmt))
      return false;

    return BindReplacementContext(Node, nullptr);
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
  const auto *CommaLHS = Result.Nodes.getNodeAs<BinaryOperator>("commaLHS");
  assert(MemcpyCall);
  assert(DstExpr);
  assert(SrcExpr);
  assert(ReplacementRoot);

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
    if (CommaLHS)
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
