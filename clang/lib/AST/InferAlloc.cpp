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
#include "llvm/ADT/STLFunctionalExtras.h"
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

namespace {
/// Strips the wrappers a placement argument may have around the expression
/// that produced the storage: parentheses, casts, and non-static member
/// accesses of a by-value result (e.g. the pointer member of a size-returning
/// allocation result). Anything that follows a pointer into different storage
/// (`->`, `*`) or names a different object (`&`) stops the walk.
const Expr *stripToStorageSource(const Expr *E) {
  while (true) {
    E = E->IgnoreParenCasts();
    const auto *ME = dyn_cast<MemberExpr>(E);
    if (!ME || ME->isArrow() || !isa<FieldDecl>(ME->getMemberDecl()))
      return E;
    E = ME->getBase();
  }
}

/// Finds the allocation call an initializer or assigned value consists of,
/// looking through parentheses, casts, temporaries, and the trivial copy or
/// move a class-typed value may carry.
const CallExpr *initializingCall(const Expr *Init) {
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

/// Returns true if \p Init is a null pointer constant (nullptr, NULL, 0).
bool isNullPointerInit(const Expr *Init) {
  Init = Init->IgnoreParenCasts();
  if (isa<CXXNullPtrLiteralExpr, GNUNullExpr>(Init))
    return true;
  const auto *IL = dyn_cast<IntegerLiteral>(Init);
  return IL && IL->getValue() == 0;
}

/// Returns the local variable an lvalue expression writes to: the variable
/// itself, or the variable whose non-static field is written (`res.p = ...`).
/// \p Direct is set when the variable itself is the lvalue.
const VarDecl *writtenVar(const Expr *E, bool &Direct) {
  E = E->IgnoreParens();
  Direct = true;
  while (const auto *ME = dyn_cast<MemberExpr>(E)) {
    if (ME->isArrow() || !isa<FieldDecl>(ME->getMemberDecl()))
      return nullptr;
    E = ME->getBase()->IgnoreParenImpCasts();
    Direct = false;
  }
  if (const auto *DRE = dyn_cast<DeclRefExpr>(E))
    if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
        VD && VD->hasLocalStorage())
      return VD;
  return nullptr;
}

/// Records, per allocation call, the type placement-constructed into its
/// result.
///
/// A placement argument that names a local variable (not a parameter, whose
/// incoming value is a definition that cannot be seen) is resolved only if the
/// function body contains exactly one definition of that variable (its
/// initializer, or one assignment; a null pointer initializer does not count)
/// and no other write to it: no compound assignment, increment, address-of,
/// write to one of its fields, non-const member call, or binding to a
/// non-const reference. Every value such a variable ever holds then comes from
/// that one definition, so the attribution holds under any control flow,
/// including loops, lambdas, and exceptions. Variables with more than one
/// definition or any other write are not used for inference.
class PlacementNewCollector : public ConstDynamicRecursiveASTVisitor {
public:
  explicit PlacementNewCollector(
      llvm::DenseMap<const CallExpr *, QualType> &CallTypes)
      : CallTypes(CallTypes) {}

  void finish() {
    for (const auto &[VD, T] : Uses)
      if (const CallExpr *CE = Defs.lookup(VD))
        record(CE, T);
  }

  bool VisitVarDecl(const VarDecl *VD) override {
    if (!VD->hasInit())
      return true;
    if (VD->hasLocalStorage() && !isNullPointerInit(VD->getInit()))
      addDef(VD, VD->getInit());
    // `T &r = p;` lets the variable be written through the reference.
    if (VD->getType()->isReferenceType())
      noteWrite(VD->getInit());
    return true;
  }

  bool VisitBinaryOperator(const BinaryOperator *BO) override {
    if (!BO->isAssignmentOp())
      return true;
    bool Direct = false;
    if (const VarDecl *VD = writtenVar(BO->getLHS(), Direct))
      addDef(VD,
             Direct && BO->getOpcode() == BO_Assign ? BO->getRHS() : nullptr);
    return true;
  }

  bool VisitUnaryOperator(const UnaryOperator *UO) override {
    // The variable may be written through the pointer, or is modified.
    if (UO->getOpcode() == UO_AddrOf || UO->isIncrementDecrementOp())
      noteWrite(UO->getSubExpr());
    return true;
  }

  bool VisitCallExpr(const CallExpr *CE) override {
    // Arguments bound to non-const references may be written by the callee.
    if (const FunctionDecl *FD = CE->getDirectCallee())
      noteReferenceArgs(FD, CE->arguments());
    return true;
  }

  bool VisitCXXConstructExpr(const CXXConstructExpr *CE) override {
    noteReferenceArgs(CE->getConstructor(), CE->arguments());
    return true;
  }

  bool VisitCXXMemberCallExpr(const CXXMemberCallExpr *MCE) override {
    // A non-const member function may modify a class-typed variable.
    const CXXMethodDecl *MD = MCE->getMethodDecl();
    if (MD && !MD->isConst() && !MD->isStatic())
      noteWrite(MCE->getImplicitObjectArgument());
    return true;
  }

  bool VisitCXXNewExpr(const CXXNewExpr *NE) override {
    // Only the reserved non-allocating placement form, new (storage) T, is
    // guaranteed to construct into its single placement argument. A
    // class-specific or otherwise user-provided operator new(size_t, void *)
    // may treat the argument as allocator state and return other storage.
    // Dependent new-expressions (e.g. in a generic lambda) have no resolved
    // operator and are skipped.
    const FunctionDecl *OperatorNew = NE->getOperatorNew();
    if (NE->getNumPlacementArgs() != 1 || !OperatorNew ||
        !OperatorNew->isReservedGlobalPlacementOperator())
      return true;
    const Expr *Source = stripToStorageSource(NE->getPlacementArg(0));
    QualType T = NE->getAllocatedType();
    if (const auto *CE = dyn_cast<CallExpr>(Source))
      record(CE, T);
    else if (const auto *DRE = dyn_cast<DeclRefExpr>(Source))
      // A parameter's incoming value is a definition that cannot be seen.
      if (const auto *VD = dyn_cast<VarDecl>(DRE->getDecl());
          VD && VD->hasLocalStorage() && !isa<ParmVarDecl>(VD))
        Uses.emplace_back(VD, T);
    return true;
  }

private:
  /// Records a definition of \p VD with \p Value (null for a write that is not
  /// a plain assignment). A second definition disqualifies the variable.
  void addDef(const VarDecl *VD, const Expr *Value) {
    auto [It, Inserted] =
        Defs.try_emplace(VD, Value ? initializingCall(Value) : nullptr);
    if (!Inserted)
      It->second = nullptr;
  }

  void noteWrite(const Expr *E) {
    bool Direct = false;
    if (const VarDecl *VD = writtenVar(E, Direct))
      addDef(VD, nullptr);
  }

  void noteReferenceArgs(const FunctionDecl *FD,
                         CallExpr::const_arg_range Args) {
    unsigned I = 0;
    for (const Expr *Arg : Args) {
      if (I >= FD->getNumParams())
        break;
      QualType ParamTy = FD->getParamDecl(I++)->getType();
      if (ParamTy->isLValueReferenceType() &&
          !ParamTy.getNonReferenceType().isConstQualified())
        noteWrite(Arg);
    }
  }

  void record(const CallExpr *CE, QualType T) {
    auto [It, Inserted] = CallTypes.try_emplace(CE, T);
    // Different types constructed into the same storage: ambiguous.
    if (!Inserted && !It->second.isNull() &&
        It->second.getCanonicalType() != T.getCanonicalType())
      It->second = QualType();
  }

  llvm::DenseMap<const CallExpr *, QualType> &CallTypes;
  // Variable -> its single defining allocation call; null once the variable
  // has a second definition or another write.
  llvm::DenseMap<const VarDecl *, const CallExpr *> Defs;
  llvm::SmallVector<std::pair<const VarDecl *, QualType>> Uses;
};
} // namespace

llvm::DenseMap<const CallExpr *, QualType>
infer_alloc::collectPlacementNewTypes(const Stmt *Body) {
  llvm::DenseMap<const CallExpr *, QualType> CallTypes;
  if (Body) {
    PlacementNewCollector Collector(CallTypes);
    Collector.TraverseStmt(Body);
    Collector.finish();
  }
  return CallTypes;
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
