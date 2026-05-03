//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UseBitCastCheck.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclFriend.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/Lex/Lexer.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/FormatVariadic.h"

using namespace clang::ast_matchers;

namespace clang::tidy::modernize {

static bool isSupportedMemcpyObjectExpr(const Expr *ExprNode) {
  ExprNode = ExprNode->IgnoreParenImpCasts();

  if (isa<DeclRefExpr>(ExprNode))
    return true;

  if (const auto *MemberPointer = dyn_cast<BinaryOperator>(ExprNode))
    return MemberPointer->isPtrMemOp() &&
           isSupportedMemcpyObjectExpr(MemberPointer->getLHS());

  if (const auto *Member = dyn_cast<MemberExpr>(ExprNode))
    return isa<FieldDecl>(Member->getMemberDecl()) &&
           isSupportedMemcpyObjectExpr(Member->getBase());

  return false;
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
         !Type->isAnyPointerType() && Type.isTriviallyCopyableType(Context) &&
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

static QualType getUnqualifiedCanonicalNonReferenceType(QualType Type) {
  if (Type.isNull())
    return {};
  return Type.getCanonicalType().getNonReferenceType().getUnqualifiedType();
}

static bool isSameUnqualifiedCanonicalType(QualType LHS, QualType RHS) {
  return LHS.getCanonicalType().getUnqualifiedType() ==
         RHS.getCanonicalType().getUnqualifiedType();
}

static bool isSameOrDerivedFrom(QualType Type, QualType Other) {
  Type = getUnqualifiedCanonicalNonReferenceType(Type);
  Other = getUnqualifiedCanonicalNonReferenceType(Other);
  if (Type == Other)
    return true;

  const auto *Record = Type->getAsCXXRecordDecl();
  const auto *OtherRecord = Other->getAsCXXRecordDecl();
  return Record && OtherRecord && Record->hasDefinition() &&
         OtherRecord->hasDefinition() && Record->isDerivedFrom(OtherRecord);
}

// This is only a cheap candidate search for preserving comma behavior in
// fix-its. It intentionally does not run overload resolution for the
// synthesized replacement expression.
static bool canBindCommaOperand(QualType OperandType, QualType ParamType) {
  OperandType = getUnqualifiedCanonicalNonReferenceType(OperandType);
  ParamType = getUnqualifiedCanonicalNonReferenceType(ParamType);
  if (OperandType.isNull() || ParamType.isNull())
    return false;

  if (OperandType->isDependentType() || ParamType->isDependentType())
    return true;

  if (isSameOrDerivedFrom(OperandType, ParamType))
    return true;

  if (OperandType->isArithmeticType() && ParamType->isArithmeticType())
    return true;

  if (OperandType->isIntegralOrEnumerationType() &&
      ParamType->isIntegralOrEnumerationType())
    return true;

  return OperandType->isAnyPointerType() && ParamType->isAnyPointerType();
}

static bool isPotentialCommaOperatorForTypes(const FunctionDecl *Function,
                                             QualType LHS, QualType RHS) {
  if (!Function || Function->getOverloadedOperator() != OO_Comma)
    return false;

  if (isa<CXXMethodDecl>(Function))
    return false;

  if (Function->getNumParams() != 2)
    return true;

  return canBindCommaOperand(LHS, Function->getParamDecl(0)->getType()) &&
         canBindCommaOperand(RHS, Function->getParamDecl(1)->getType());
}

static bool isPotentialCommaOperatorForTypes(const NamedDecl *Decl,
                                             QualType LHS, QualType RHS) {
  if (!Decl)
    return false;

  if (const auto *FunctionTemplate = dyn_cast<FunctionTemplateDecl>(Decl))
    return isPotentialCommaOperatorForTypes(
        FunctionTemplate->getTemplatedDecl(), LHS, RHS);
  return isPotentialCommaOperatorForTypes(Decl->getAsFunction(), LHS, RHS);
}

static bool hasPotentialNamespaceCommaOperator(const DeclContext *Context,
                                               const ASTContext &ASTContext,
                                               QualType LHS, QualType RHS) {
  if (!Context)
    return false;

  const DeclarationName CommaOperatorName =
      ASTContext.DeclarationNames.getCXXOperatorName(OO_Comma);
  for (; Context; Context = Context->getParent()) {
    if (!isa<NamespaceDecl, TranslationUnitDecl>(Context))
      continue;

    for (const NamedDecl *Decl :
         Context->getRedeclContext()->lookup(CommaOperatorName)) {
      if (isPotentialCommaOperatorForTypes(Decl, LHS, RHS))
        return true;
    }
  }

  return false;
}

static const CXXRecordDecl *getDefinition(const CXXRecordDecl *Record) {
  if (!Record)
    return nullptr;
  if (const auto *Definition = Record->getDefinition())
    return Definition;
  return Record;
}

static bool hasMemberCommaOperator(const CXXRecordDecl *Record) {
  Record = getDefinition(Record);
  return Record &&
         llvm::any_of(Record->methods(), [](const CXXMethodDecl *Method) {
           return Method->getOverloadedOperator() == OO_Comma;
         });
}

static bool hasPotentialFriendCommaOperator(const CXXRecordDecl *Record,
                                            QualType LHS, QualType RHS) {
  Record = getDefinition(Record);
  return Record &&
         llvm::any_of(Record->friends(), [&](const FriendDecl *Friend) {
           return isPotentialCommaOperatorForTypes(Friend->getFriendDecl(), LHS,
                                                   RHS);
         });
}

static bool mayFindAssociatedCommaOperator(QualType Type,
                                           const ASTContext &ASTContext,
                                           QualType LHS, QualType RHS,
                                           bool IncludeMemberOperators) {
  Type = getUnqualifiedCanonicalNonReferenceType(Type);
  if (Type.isNull())
    return false;

  if (const auto *Record = Type->getAsCXXRecordDecl())
    return (IncludeMemberOperators && hasMemberCommaOperator(Record)) ||
           hasPotentialFriendCommaOperator(Record, LHS, RHS) ||
           hasPotentialNamespaceCommaOperator(Record->getDeclContext(),
                                              ASTContext, LHS, RHS);

  if (const auto *Enum = Type->getAs<EnumType>())
    return hasPotentialNamespaceCommaOperator(Enum->getDecl()->getDeclContext(),
                                              ASTContext, LHS, RHS);

  return false;
}

static bool mayCallOverloadedComma(QualType AssignmentType,
                                   const Expr *CommaRHS,
                                   const ASTContext &ASTContext) {
  const QualType CommaRHSType = CommaRHS ? CommaRHS->getType() : QualType();
  return mayFindAssociatedCommaOperator(AssignmentType, ASTContext,
                                        AssignmentType, CommaRHSType,
                                        /*IncludeMemberOperators=*/true) ||
         mayFindAssociatedCommaOperator(CommaRHSType, ASTContext,
                                        AssignmentType, CommaRHSType,
                                        /*IncludeMemberOperators=*/false);
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
                                          const BinaryOperator *CommaContext) {
    Builder->setBinding("replacementRoot",
                        DynTypedNode::create(ReplacementRoot));
    if (CommaContext)
      Builder->setBinding("commaContext", DynTypedNode::create(*CommaContext));
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
  const auto *CommaContext =
      Result.Nodes.getNodeAs<BinaryOperator>("commaContext");
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
  const std::string Replacement = [&]() -> std::string {
    std::string Assignment = llvm::formatv("{0} = std::bit_cast<{1}>({2})",
                                           DstText, DstTypeName, SrcText)
                                 .str();
    if (CommaContext && mayCallOverloadedComma(DstType, CommaContext->getRHS(),
                                               *Result.Context))
      return llvm::formatv("(void)({0})", Assignment).str();
    return Assignment;
  }();

  const DiagnosticBuilder Diag =
      diag(MemcpyCall->getBeginLoc(),
           "use 'std::bit_cast' instead of 'memcpy' for type punning");
  Diag << FixItHint::CreateReplacement(ReplacementRoot->getSourceRange(),
                                       Replacement);
  Diag << IncludeInserter.createIncludeInsertion(
      SM.getFileID(MemcpyCall->getBeginLoc()), "<bit>");
}

} // namespace clang::tidy::modernize
