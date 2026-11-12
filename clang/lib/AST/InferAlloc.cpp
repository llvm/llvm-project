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
#include <vector>

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
                                        const CastExpr *CastE) {
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
/// looking through casts and the temporaries and trivial copies a class-typed
/// value may carry.
const CallExpr *initializingCall(const Expr *Init) {
  while (Init) {
    Init = Init->IgnoreParenCasts();
    if (const auto *EWC = dyn_cast<ExprWithCleanups>(Init)) {
      Init = EWC->getSubExpr();
    } else if (const auto *MTE = dyn_cast<MaterializeTemporaryExpr>(Init)) {
      Init = MTE->getSubExpr();
    } else if (const auto *BTE = dyn_cast<CXXBindTemporaryExpr>(Init)) {
      Init = BTE->getSubExpr();
    } else if (const auto *CE = dyn_cast<CXXConstructExpr>(Init);
               CE && CE->getNumArgs() == 1 &&
               CE->getConstructor()->isCopyOrMoveConstructor() &&
               CE->getConstructor()->isTrivial()) {
      // A non-trivial copy or move may not be elided (-fno-elide-constructors)
      // and could replace the pointer.
      Init = CE->getArg(0);
    } else if (const auto *FCE = dyn_cast<CXXFunctionalCastExpr>(Init)) {
      Init = FCE->getSubExpr();
    } else {
      break;
    }
  }
  return dyn_cast_or_null<CallExpr>(Init);
}

/// Records, per allocation call, the type placement-constructed into its
/// result.
///
/// Placement arguments that name a variable are resolved with a lexical
/// def-use chain: a use `new (p) T` is matched to the nearest preceding
/// definition of `p` (its initializer or an assignment `p = <call>`) in
/// traversal order, provided that definition lexically dominates the use,
/// i.e. it is in the same control-flow context as the use or an enclosing
/// one. A definition in a branch that does not enclose the use, a later
/// definition that can reach the use around a loop, or any other write to
/// the variable (compound assignment, increment, address-of) makes the use
/// ambiguous and nothing is inferred for it.
class PlacementNewCollector : public ConstDynamicRecursiveASTVisitor {
public:
  explicit PlacementNewCollector(
      llvm::DenseMap<const CallExpr *, QualType> &CallTypes)
      : CallTypes(CallTypes) {
    // Context 0 is the function body itself.
    Contexts.push_back({/*Parent=*/-1, /*IsLoop=*/false});
  }

  void finish() {
    for (const Use &U : Uses) {
      const CallExpr *CE = resolve(U);
      if (CE)
        record(CE, U.Type);
    }
  }

  bool VisitVarDecl(const VarDecl *VD) override {
    if (VD->isLocalVarDecl())
      addDef(VD, initializingCall(VD->getInit()));
    return true;
  }

  bool VisitBinaryOperator(const BinaryOperator *BO) override {
    if (!BO->isAssignmentOp())
      return true;
    if (const VarDecl *VD = referencedVar(BO->getLHS()))
      addDef(VD, BO->getOpcode() == BO_Assign ? initializingCall(BO->getRHS())
                                              : nullptr);
    return true;
  }

  bool VisitUnaryOperator(const UnaryOperator *UO) override {
    if (UO->getOpcode() != UO_AddrOf && !UO->isIncrementDecrementOp())
      return true;
    // The variable may be written through the pointer or reference.
    if (const VarDecl *VD = referencedVar(UO->getSubExpr()))
      addDef(VD, nullptr);
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
    else if (const VarDecl *VD = referencedVar(Source))
      Uses.push_back({VD, Seq++, CurContext, T});
    return true;
  }

  // Control-flow constructs open a new context for the statements they
  // contain; a definition inside one does not dominate uses outside it.
#define PLACEMENT_CONTEXT(CLASS, IS_LOOP)                                      \
  bool Traverse##CLASS(const CLASS *S) override {                              \
    return inContext(IS_LOOP, [&] {                                            \
      return ConstDynamicRecursiveASTVisitor::Traverse##CLASS(S);              \
    });                                                                        \
  }
  PLACEMENT_CONTEXT(IfStmt, false)
  PLACEMENT_CONTEXT(SwitchStmt, false)
  PLACEMENT_CONTEXT(ConditionalOperator, false)
  PLACEMENT_CONTEXT(BinaryConditionalOperator, false)
  PLACEMENT_CONTEXT(LambdaExpr, false)
  PLACEMENT_CONTEXT(StmtExpr, false)
  PLACEMENT_CONTEXT(ForStmt, true)
  PLACEMENT_CONTEXT(WhileStmt, true)
  PLACEMENT_CONTEXT(DoStmt, true)
  PLACEMENT_CONTEXT(CXXForRangeStmt, true)
#undef PLACEMENT_CONTEXT

private:
  struct Context {
    int Parent;
    bool IsLoop;
  };
  struct Def {
    unsigned Seq;
    int Context;
    const CallExpr *Call; // null: a write that is not an allocation call
  };
  struct Use {
    const VarDecl *Var;
    unsigned Seq;
    int Context;
    QualType Type;
  };

  bool inContext(bool IsLoop, llvm::function_ref<bool()> Traverse) {
    int Saved = CurContext;
    Contexts.push_back({Saved, IsLoop});
    CurContext = Contexts.size() - 1;
    bool Result = Traverse();
    CurContext = Saved;
    return Result;
  }

  static const VarDecl *referencedVar(const Expr *E) {
    if (const auto *DRE = dyn_cast<DeclRefExpr>(E->IgnoreParenImpCasts()))
      return dyn_cast<VarDecl>(DRE->getDecl());
    return nullptr;
  }

  void addDef(const VarDecl *VD, const CallExpr *CE) {
    Defs[VD].push_back({Seq++, CurContext, CE});
  }

  bool isAncestorOrSelf(int Ancestor, int Ctx) const {
    for (; Ctx >= 0; Ctx = Contexts[Ctx].Parent)
      if (Ctx == Ancestor)
        return true;
    return false;
  }

  bool isInLoop(int Ctx) const {
    for (; Ctx >= 0; Ctx = Contexts[Ctx].Parent)
      if (Contexts[Ctx].IsLoop)
        return true;
    return false;
  }

  /// Returns the allocation call that defines the variable of \p U at the
  /// use, or null if there is none or it cannot be determined.
  const CallExpr *resolve(const Use &U) const {
    auto It = Defs.find(U.Var);
    if (It == Defs.end())
      return nullptr;
    const Def *Reaching = nullptr;
    bool LaterDef = false;
    for (const Def &D : It->second) {
      if (D.Seq < U.Seq)
        Reaching = &D; // defs are in traversal order; keep the last one
      else
        LaterDef = true;
    }
    if (!Reaching || !Reaching->Call ||
        !isAncestorOrSelf(Reaching->Context, U.Context))
      return nullptr;
    // A later definition can reach the use around a loop back edge.
    if (LaterDef && isInLoop(U.Context))
      return nullptr;
    return Reaching->Call;
  }

  void record(const CallExpr *CE, QualType T) {
    auto [It, Inserted] = CallTypes.try_emplace(CE, T);
    // Different types constructed into the same storage: ambiguous.
    if (!Inserted && !It->second.isNull() &&
        It->second.getCanonicalType() != T.getCanonicalType())
      It->second = QualType();
  }

  llvm::DenseMap<const CallExpr *, QualType> &CallTypes;
  std::vector<Context> Contexts;
  int CurContext = 0;
  unsigned Seq = 0;
  llvm::DenseMap<const VarDecl *, llvm::SmallVector<Def, 2>> Defs;
  std::vector<Use> Uses;
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
