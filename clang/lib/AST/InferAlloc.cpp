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
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Type.h"
#include "clang/Basic/IdentifierTable.h"
#include "llvm/ADT/SmallPtrSet.h"

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

QualType infer_alloc::inferPossibleType(const CallExpr *E,
                                        const ASTContext &Ctx,
                                        const CastExpr *CastE,
                                        PlacementNewTypes *PlacementTypes) {
  QualType AllocType;
  // First check arguments.
  for (const Expr *Arg : E->arguments()) {
    AllocType = inferPossibleTypeFromArithSizeofExpr(Arg);
    if (AllocType.isNull())
      AllocType = inferPossibleTypeFromVarInitSizeofExpr(Arg);
    if (!AllocType.isNull())
      break;
  }
  // Then check later casts.
  if (AllocType.isNull())
    AllocType = inferPossibleTypeFromCastExpr(E, CastE);
  // Finally check if the result is used as placement-new storage.
  if (AllocType.isNull() && PlacementTypes)
    AllocType = PlacementTypes->lookup(E);
  return AllocType;
}

namespace {
/// Strips the wrappers a placement argument may have around the expression
/// that produced the storage: parentheses, casts, member accesses (e.g. the
/// pointer member of a size-returning allocation result), and
/// address-of/dereference pairs.
const Expr *stripToStorageSource(const Expr *E) {
  while (true) {
    E = E->IgnoreParenCasts();
    if (const auto *ME = dyn_cast<MemberExpr>(E)) {
      E = ME->getBase();
      continue;
    }
    if (const auto *UO = dyn_cast<UnaryOperator>(E);
        UO && (UO->getOpcode() == UO_AddrOf || UO->getOpcode() == UO_Deref)) {
      E = UO->getSubExpr();
      continue;
    }
    return E;
  }
}

/// Finds the allocation call that initializes a variable, looking through the
/// temporaries and elidable copies a class-typed initializer may carry.
const CallExpr *initializingCall(const Expr *Init) {
  while (Init) {
    Init = Init->IgnoreParenImpCasts();
    if (const auto *EWC = dyn_cast<ExprWithCleanups>(Init)) {
      Init = EWC->getSubExpr();
    } else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Init)) {
      Init = MTE->getSubExpr();
    } else if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(Init)) {
      Init = BTE->getSubExpr();
    } else if (const auto *CE = dyn_cast<CXXConstructExpr>(Init);
               CE && CE->getNumArgs() == 1 &&
               (CE->isElidable() ||
                CE->getConstructor()->isCopyOrMoveConstructor())) {
      Init = CE->getArg(0);
    } else if (const auto *FCE = dyn_cast<CXXFunctionalCastExpr>(Init)) {
      Init = FCE->getSubExpr();
    } else {
      break;
    }
  }
  return dyn_cast_or_null<CallExpr>(Init);
}

/// Records placement-new types per allocation call and per variable.
class PlacementNewCollector
    : public RecursiveASTVisitor<PlacementNewCollector> {
public:
  PlacementNewCollector(
      llvm::DenseMap<const CallExpr *, QualType> &CallTypes,
      llvm::DenseMap<const VarDecl *, QualType> &VarTypes,
      llvm::DenseMap<const CallExpr *, const VarDecl *> &CallVars)
      : CallTypes(CallTypes), VarTypes(VarTypes), CallVars(CallVars) {}

  bool shouldVisitTemplateInstantiations() const { return true; }
  bool shouldVisitImplicitCode() const { return true; }

  bool VisitVarDecl(const VarDecl *VD) {
    if (const CallExpr *CE = initializingCall(VD->getInit()))
      CallVars.try_emplace(CE, VD);
    return true;
  }

  bool VisitCXXNewExpr(const CXXNewExpr *NE) {
    // Only the standard non-allocating placement form, new (storage) T, takes
    // the storage as its single placement argument.
    if (NE->getNumPlacementArgs() != 1)
      return true;
    const Expr *Source = stripToStorageSource(NE->getPlacementArg(0));
    QualType T = NE->getAllocatedType();
    if (const auto *CE = dyn_cast<CallExpr>(Source))
      record(CallTypes, CE, T);
    else if (const auto *DRE = dyn_cast<DeclRefExpr>(Source))
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl()))
        record(VarTypes, VD, T);
    return true;
  }

private:
  template <typename Key>
  static void record(llvm::DenseMap<Key, QualType> &Map, Key K, QualType T) {
    auto [It, Inserted] = Map.try_emplace(K, T);
    // Different types constructed into the same storage: ambiguous.
    if (!Inserted && !It->second.isNull() &&
        It->second.getCanonicalType() != T.getCanonicalType())
      It->second = QualType();
  }

  llvm::DenseMap<const CallExpr *, QualType> &CallTypes;
  llvm::DenseMap<const VarDecl *, QualType> &VarTypes;
  llvm::DenseMap<const CallExpr *, const VarDecl *> &CallVars;
};
} // namespace

void PlacementNewTypes::collect() {
  Collected = true;
  if (!Body)
    return;
  PlacementNewCollector Collector(CallTypes, VarTypes, CallVars);
  Collector.TraverseStmt(const_cast<Stmt *>(Body));
}

QualType PlacementNewTypes::lookup(const CallExpr *E) {
  if (!Collected)
    collect();
  if (auto It = CallTypes.find(E); It != CallTypes.end())
    return It->second;
  if (auto VarIt = CallVars.find(E); VarIt != CallVars.end())
    if (auto It = VarTypes.find(VarIt->second); It != VarTypes.end())
      return It->second;
  return QualType();
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
