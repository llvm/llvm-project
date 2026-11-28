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
#include "clang/Tooling/FixIt.h"
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
  return !Type.isNull() && Type.isTriviallyCopyableType(Context);
}

static bool canAssignBitCastResult(QualType Type) {
  Type = Type.getCanonicalType().getNonReferenceType();
  if (Type.isConstQualified() || Type->isArrayType() ||
      (Type.isVolatileQualified() && Type->isRecordType()))
    return false;

  const auto *Record = Type->getAsCXXRecordDecl();
  return !Record || Record->hasSimpleCopyAssignment() ||
         Record->hasSimpleMoveAssignment();
}

static bool isSameUnqualifiedCanonicalType(QualType LHS, QualType RHS) {
  return LHS.getCanonicalType().getUnqualifiedType() ==
         RHS.getCanonicalType().getUnqualifiedType();
}

static bool isUnnameableType(QualType Type) {
  const TagDecl *Tag = Type->getAsTagDecl();
  return Tag && !Tag->getIdentifier() && !Tag->getTypedefNameForAnonDecl();
}

static bool canUseDecltypeAsBitCastType(const Expr *DstExpr) {
  if (const auto *Ref = dyn_cast<DeclRefExpr>(DstExpr))
    if (const auto *Var = dyn_cast<VarDecl>(Ref->getDecl()))
      return !Var->getType()->isReferenceType();

  return isa<MemberExpr>(DstExpr);
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
      .Case([&](const CompoundStmt *Block) {
        return llvm::any_of(Block->body(), IsCurrentBody);
      })
      .Case([&](const IfStmt *If) {
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
  const CastExpr *DirectVoidCast = nullptr;
  const BinaryOperator *CommaContext = nullptr;
  const BinaryOperator *OverloadableCommaContext = nullptr;
  const auto IsTransparentReplacementParent = [](const Expr *ExprNode) {
    return isa<ExprWithCleanups, ImplicitCastExpr, MaterializeTemporaryExpr,
               CXXBindTemporaryExpr, ParenExpr>(ExprNode);
  };
  const auto BindReplacementContext = [&](const Expr &ReplacementRoot) {
    Builder->setBinding("replacementRoot",
                        DynTypedNode::create(ReplacementRoot));
    if (CommaContext)
      Builder->setBinding("commaContext", DynTypedNode::create(*CommaContext));
    if (OverloadableCommaContext)
      Builder->setBinding("overloadableCommaContext",
                          DynTypedNode::create(*OverloadableCommaContext));
    return true;
  };
  const auto RecordCommaContext = [&](const BinaryOperator *Comma,
                                      const Expr *Sibling) {
    CommaContext = Comma;
    const QualType SiblingType = Sibling->getType();
    if (Sibling->isTypeDependent() || SiblingType.isNull() ||
        SiblingType->isOverloadableType())
      OverloadableCommaContext = Comma;
  };
  const auto IsCurrentOperand = [&](const Expr *Operand) {
    // The traversal can skip parentheses that the BinaryOperator still owns.
    return Operand == Current || Operand->IgnoreParenImpCasts() == Current;
  };

  while (true) {
    const auto Parents = Finder->getASTContext().getParents(*Current);
    if (Parents.size() != 1)
      return false;

    if (DirectVoidCast) {
      if (const auto *ParentExpr = Parents[0].get<Expr>()) {
        if (IsTransparentReplacementParent(ParentExpr)) {
          Current = ParentExpr;
          continue;
        }
      } else if (const auto *ParentStmt = Parents[0].get<Stmt>();
                 ParentStmt && isStatementBody(Current, ParentStmt)) {
        return BindReplacementContext(*DirectVoidCast);
      }

      Builder->setBinding("preservedVoidCast",
                          DynTypedNode::create(*DirectVoidCast));
      return BindReplacementContext(Node);
    }

    if (const auto *ParentExpr = Parents[0].get<Expr>()) {
      if (IsTransparentReplacementParent(ParentExpr)) {
        Current = ParentExpr;
        continue;
      }

      if (const auto *Cast = dyn_cast<CastExpr>(ParentExpr)) {
        if (Cast->getCastKind() != CK_ToVoid)
          return false;
        if (!SawDiscardedCommaRHS)
          DirectVoidCast = Cast;

        Current = Cast;
        continue;
      }

      const auto *Comma = dyn_cast<BinaryOperator>(ParentExpr);
      if (!Comma || Comma->getOpcode() != BO_Comma)
        return false;
      if (IsCurrentOperand(Comma->getLHS())) {
        RecordCommaContext(Comma, Comma->getRHS());
        return BindReplacementContext(Node);
      }
      if (!IsCurrentOperand(Comma->getRHS()))
        return false;

      // A memcpy on the right-hand side of `,` is safe only if the enclosing
      // comma expression is itself discarded, so keep walking from the comma
      // node. Remember every sibling because changing the memcpy result type
      // can make an overloaded comma viable at any level.
      RecordCommaContext(Comma, Comma->getLHS());
      SawDiscardedCommaRHS = true;
      Current = Comma;
      continue;
    }

    const auto *ParentStmt = Parents[0].get<Stmt>();
    if (!ParentStmt || !isStatementBody(Current, ParentStmt))
      return false;

    return BindReplacementContext(Node);
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

AST_MATCHER(FunctionDecl, hasSizeTypeThirdParameter) {
  const auto *Type = Node.getType()->getAs<FunctionProtoType>();
  return Type && Type->getNumParams() == 3 &&
         ASTContext::hasSameType(Type->getParamType(2),
                                 Finder->getASTContext().getSizeType());
}

} // namespace

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
  const auto MemcpyDecl = functionDecl(
      hasAnyName("::memcpy", "::std::memcpy"), parameterCountIs(3),
      returns(pointerType(pointee(voidType()))),
      hasParameter(0, hasType(pointerType(pointee(voidType())))),
      hasParameter(1, hasType(pointerType(
                          pointee(qualType(isConstQualified(), voidType()))))),
      hasSizeTypeThirdParameter());
  Finder->addMatcher(callExpr(callee(MemcpyDecl), isBitCastMemcpyCandidate(),
                              hasBitCastReplacementContext())
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
  const auto *OverloadableCommaContext =
      Result.Nodes.getNodeAs<BinaryOperator>("overloadableCommaContext");
  const auto *PreservedVoidCast =
      Result.Nodes.getNodeAs<CastExpr>("preservedVoidCast");
  assert(MemcpyCall && "memcpy call must be bound");
  assert(DstExpr && "destination expression must be bound");
  assert(SrcExpr && "source expression must be bound");
  assert(ReplacementRoot && "replacement root must be bound");

  const SourceManager &SM = *Result.SourceManager;
  StringRef DstText = tooling::fixit::getText(*DstExpr, *Result.Context);
  StringRef SrcText = tooling::fixit::getText(*SrcExpr, *Result.Context);
  if (DstText.empty() || SrcText.empty())
    return;

  const PrintingPolicy &Policy = Result.Context->getPrintingPolicy();
  const QualType DstType =
      DstExpr->getType().getNonReferenceType().getUnqualifiedType();
  const bool UseDecltype = isUnnameableType(DstType);
  if (UseDecltype && !canUseDecltypeAsBitCastType(DstExpr)) {
    diag(MemcpyCall->getBeginLoc(),
         "use 'std::bit_cast' instead of 'memcpy' for type punning");
    return;
  }
  const std::string DstTypeName =
      UseDecltype ? llvm::formatv("decltype({0})", DstText).str()
                  : DstType.getAsString(Policy);
  const std::string Replacement = [&]() -> std::string {
    std::string Assignment = llvm::formatv("{0} = std::bit_cast<{1}>({2})",
                                           DstText, DstTypeName, SrcText)
                                 .str();
    if (PreservedVoidCast)
      return llvm::formatv("({0})", Assignment).str();
    if (CommaContext &&
        (OverloadableCommaContext || DstType->isOverloadableType()))
      return llvm::formatv("(void)({0})", Assignment).str();
    return Assignment;
  }();

  const DiagnosticBuilder Diag =
      diag(MemcpyCall->getBeginLoc(),
           "use 'std::bit_cast' instead of 'memcpy' for type punning");
  Diag << tooling::fixit::createReplacement(*ReplacementRoot, Replacement);
  Diag << IncludeInserter.createIncludeInsertion(
      SM.getFileID(MemcpyCall->getBeginLoc()), "<bit>");
}

} // namespace clang::tidy::modernize
