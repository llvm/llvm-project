//===--- InferAlloc.cpp - Allocation type inference -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements allocation-related type inference.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/InferAlloc.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DynamicRecursiveASTVisitor.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace clang;
using namespace infer_alloc;

static bool
typeContainsPointer(QualType T,
                    llvm::SmallPtrSet<const RecordDecl *, 4> &VisitedRD,
                    bool &IncompleteType) {
  QualType CanonicalType = T.getCanonicalType();
  if (CanonicalType->isAnyPointerType() || CanonicalType->isReferenceType() ||
      CanonicalType->isMemberFunctionPointerType() ||
      CanonicalType->isBlockPointerType())
    return true; // base case

  // Look through typedef chain to check for special types.
  for (QualType CurrentT = T; const auto *TT = CurrentT->getAs<TypedefType>();
       CurrentT = TT->getDecl()->getUnderlyingType()) {
    const IdentifierInfo *II = TT->getDecl()->getIdentifier();
    // Special Case: Syntactically uintptr_t is not a pointer; semantically,
    // however, very likely used as such. Therefore, classify uintptr_t as a
    // pointer, too.
    if (II && II->isStr("uintptr_t"))
      return true;
  }

  // The type is an array; check the element type.
  if (const ArrayType *AT = dyn_cast<ArrayType>(CanonicalType))
    return typeContainsPointer(AT->getElementType(), VisitedRD, IncompleteType);

  // The type is an atomic type.
  if (const AtomicType *AT = dyn_cast<AtomicType>(CanonicalType))
    return typeContainsPointer(AT->getValueType(), VisitedRD, IncompleteType);

  // The type is a struct, class, or union.
  if (const RecordDecl *RD = CanonicalType->getAsRecordDecl()) {
    if (!RD->isCompleteDefinition()) {
      IncompleteType = true;
      return false;
    }
    if (!VisitedRD.insert(RD).second)
      return false; // already visited
    // Check all fields.
    for (const FieldDecl *Field : RD->fields()) {
      if (typeContainsPointer(Field->getType(), VisitedRD, IncompleteType))
        return true;
    }
    // For C++ classes, also check base classes.
    if (const CXXRecordDecl *CXXRD = dyn_cast<CXXRecordDecl>(RD)) {
      // Polymorphic types require a vptr.
      if (CXXRD->isDynamicClass())
        return true;
      for (const CXXBaseSpecifier &Base : CXXRD->bases()) {
        if (typeContainsPointer(Base.getType(), VisitedRD, IncompleteType))
          return true;
      }
    }
  }
  return false;
}

/// Infer type from a simple sizeof expression.
static QualType inferTypeFromSizeofExpr(const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  if (const auto *UET = dyn_cast<UnaryExprOrTypeTraitExpr>(Arg)) {
    if (UET->getKind() == UETT_SizeOf) {
      if (UET->isArgumentType())
        return UET->getArgumentTypeInfo()->getType();
      else
        return UET->getArgumentExpr()->getType();
    }
  }
  return QualType();
}

/// Infer type from an arithmetic expression involving a sizeof. For example:
///
///   malloc(sizeof(MyType) + padding);  // infers 'MyType'
///   malloc(sizeof(MyType) * 32);       // infers 'MyType'
///   malloc(32 * sizeof(MyType));       // infers 'MyType'
///   malloc(sizeof(MyType) << 1);       // infers 'MyType'
///   ...
///
/// More complex arithmetic expressions are supported, but are a heuristic, e.g.
/// when considering allocations for structs with flexible array members:
///
///   malloc(sizeof(HasFlexArray) + sizeof(int) * 32);  // infers 'HasFlexArray'
///
static QualType inferPossibleTypeFromArithSizeofExpr(const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  // The argument is a lone sizeof expression.
  if (QualType T = inferTypeFromSizeofExpr(Arg); !T.isNull())
    return T;
  if (const auto *BO = dyn_cast<BinaryOperator>(Arg)) {
    // Argument is an arithmetic expression. Cover common arithmetic patterns
    // involving sizeof.
    switch (BO->getOpcode()) {
    case BO_Add:
    case BO_Div:
    case BO_Mul:
    case BO_Shl:
    case BO_Shr:
    case BO_Sub:
      if (QualType T = inferPossibleTypeFromArithSizeofExpr(BO->getLHS());
          !T.isNull())
        return T;
      if (QualType T = inferPossibleTypeFromArithSizeofExpr(BO->getRHS());
          !T.isNull())
        return T;
      break;
    default:
      break;
    }
  }
  return QualType();
}

/// If the expression E is a reference to a variable, infer the type from a
/// variable's initializer if it contains a sizeof. Beware, this is a heuristic
/// and ignores if a variable is later reassigned. For example:
///
///   size_t my_size = sizeof(MyType);
///   void *x = malloc(my_size);  // infers 'MyType'
///
static QualType inferPossibleTypeFromVarInitSizeofExpr(const Expr *E) {
  const Expr *Arg = E->IgnoreParenImpCasts();
  if (const auto *DRE = dyn_cast<DeclRefExpr>(Arg)) {
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl())) {
      if (const Expr *Init = VD->getInit())
        return inferPossibleTypeFromArithSizeofExpr(Init);
    }
  }
  return QualType();
}

/// Deduces the allocated type by checking if the allocation call's result
/// is immediately used in a cast expression. For example:
///
///   MyType *x = (MyType *)malloc(4096);  // infers 'MyType'
///
static QualType inferPossibleTypeFromCastExpr(const CallExpr *CallE,
                                              const CastExpr *CastE) {
  if (!CastE)
    return QualType();
  QualType PtrType = CastE->getType();
  if (PtrType->isPointerType())
    return PtrType->getPointeeType();
  return QualType();
}

QualType infer_alloc::inferPossibleType(
    const CallExpr *E, const ASTContext &Ctx, const CastExpr *CastE,
    llvm::function_ref<QualType(const CallExpr *)> PlacementNewType) {
  QualType AllocType;
  // First check arguments: a sizeof states what the allocation was sized for.
  for (const Expr *Arg : E->arguments()) {
    AllocType = inferPossibleTypeFromArithSizeofExpr(Arg);
    if (AllocType.isNull())
      AllocType = inferPossibleTypeFromVarInitSizeofExpr(Arg);
    if (!AllocType.isNull())
      break;
  }
  // Then check for a placement new into the result, which names the object
  // constructed there and is stronger evidence than a cast of the result.
  if (AllocType.isNull() && PlacementNewType)
    AllocType = PlacementNewType(E);
  // Then check later casts.
  if (AllocType.isNull())
    AllocType = inferPossibleTypeFromCastExpr(E, CastE);
  return AllocType;
}

/// Strips parentheses and non-static field accesses of a by-value object
/// (e.g. the pointer member of a size-returning allocation result) from an
/// lvalue, down to the expression naming the object. Indirect member accesses
/// (`->`) name other storage and stop the walk.
static const Expr *stripFields(const Expr *E) {
  E = E->IgnoreParens();
  while (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (ME->isArrow() || !isa<FieldDecl>(ME->getMemberDecl()))
      break;
    // The base of a member of a temporary is a MaterializeTemporaryExpr.
    E = ME->getBase()->IgnoreParenImpCasts();
  }
  return E;
}

/// Finds the allocation call an initializer consists of, looking through
/// parentheses, casts, temporaries, and the trivial copy or move a class-typed
/// value may carry.
static const CallExpr *initializingCall(const Expr *Init) {
  while (Init) {
    Init = Init->IgnoreImplicit()->IgnoreParenCasts();
    const auto *CE = dyn_cast<CXXConstructExpr>(Init);
    // A non-trivial copy or move may not be elided (-fno-elide-constructors)
    // and could replace the pointer.
    if (!CE || CE->getNumArgs() != 1 ||
        !CE->getConstructor()->isCopyOrMoveConstructor() ||
        !CE->getConstructor()->isTrivial())
      break;
    Init = CE->getArg(0);
  }
  return dyn_cast_or_null<CallExpr>(Init);
}

namespace {
/// Implements collectPlacementNewTypes(). A variable qualifies if every
/// reference to it other than its initializer is an lvalue-to-rvalue read.
class PlacementNewCollector : public ConstDynamicRecursiveASTVisitor {
public:
  // Resolved after the traversal: a use that is not a read may follow the
  // placement new and still disqualify the variable.
  llvm::DenseMap<const CallExpr *, QualType> finish() && {
    for (const auto &[VD, T] : Uses)
      if (const CallExpr *CE = Defs.lookup(VD))
        record(CE, T);
    return std::move(CallTypes);
  }

  bool VisitVarDecl(const VarDecl *VD) override {
    if (VD->hasLocalStorage() && !isa<ParmVarDecl>(VD))
      if (const CallExpr *CE = initializingCall(VD->getInit()))
        Defs.try_emplace(VD, CE);
    return true;
  }

  // A read of the variable or of one of its fields. Traversal is pre-order,
  // so the conversion is seen before the DeclRefExpr below it.
  bool VisitImplicitCastExpr(const ImplicitCastExpr *ICE) override {
    if (ICE->getCastKind() == CK_LValueToRValue)
      if (const auto *DRE =
              dyn_cast<DeclRefExpr>(stripFields(ICE->getSubExpr())))
        Reads.insert(DRE);
    return true;
  }

  // Any other reference may modify the variable.
  bool VisitDeclRefExpr(const DeclRefExpr *DRE) override {
    if (!Reads.contains(DRE))
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Defs.erase(VD);
    return true;
  }

  bool VisitCXXNewExpr(const CXXNewExpr *NE) override {
    // Only the reserved placement form is known to construct into its
    // argument; dependent new-expressions have no operator new yet.
    const FunctionDecl *OperatorNew = NE->getOperatorNew();
    if (NE->getNumPlacementArgs() != 1 || !OperatorNew ||
        !OperatorNew->isReservedGlobalPlacementOperator())
      return true;
    const Expr *Source =
        stripFields(NE->getPlacementArg(0)->IgnoreParenCasts());
    QualType T = NE->getAllocatedType();
    if (const auto *CE = dyn_cast<CallExpr>(Source))
      record(CE, T);
    else if (const auto *DRE = dyn_cast<DeclRefExpr>(Source))
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        Uses.emplace_back(VD, T);
    return true;
  }

private:
  void record(const CallExpr *CE, QualType T) {
    auto [It, Inserted] = CallTypes.try_emplace(CE, T);
    // Different types constructed into the same storage: ambiguous.
    if (!Inserted && !It->second.isNull() &&
        It->second.getCanonicalType() != T.getCanonicalType())
      It->second = QualType();
  }

  llvm::DenseMap<const CallExpr *, QualType> CallTypes;
  // Variable -> the allocation call initializing it; erased on any use that
  // is not a read.
  llvm::DenseMap<const VarDecl *, const CallExpr *> Defs;
  llvm::SmallPtrSet<const DeclRefExpr *, 16> Reads;
  llvm::SmallVector<std::pair<const VarDecl *, QualType>> Uses;
};
} // namespace

llvm::DenseMap<const CallExpr *, QualType>
infer_alloc::collectPlacementNewTypes(const Stmt *Body) {
  PlacementNewCollector Collector;
  if (Body)
    Collector.TraverseStmt(Body);
  return std::move(Collector).finish();
}

std::optional<llvm::AllocTokenMetadata>
infer_alloc::getAllocTokenMetadata(QualType T, const ASTContext &Ctx) {
  llvm::AllocTokenMetadata ATMD;

  // Get unique type name.
  PrintingPolicy Policy(Ctx.getLangOpts());
  Policy.SuppressTagKeyword = true;
  Policy.FullyQualifiedName = true;
  llvm::raw_svector_ostream TypeNameOS(ATMD.TypeName);
  T.getCanonicalType().print(TypeNameOS, Policy);

  // Check if QualType contains a pointer. Implements a simple DFS to
  // recursively check if a type contains a pointer type.
  llvm::SmallPtrSet<const RecordDecl *, 4> VisitedRD;
  bool IncompleteType = false;
  ATMD.ContainsPointer = typeContainsPointer(T, VisitedRD, IncompleteType);
  if (!ATMD.ContainsPointer && IncompleteType)
    return std::nullopt;

  return ATMD;
}
