//===- SymbolManager.h - Management of Symbolic Values ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymbolManager, a class that manages symbolic values
//  created for use by ExprEngine and related classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMBOLMANAGER_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMBOLMANAGER_H

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Analysis/AnalysisDeclContext.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntPtr.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/StoreRef.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableSet.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Allocator.h"
#include <cassert>

namespace clang {

class ASTContext;
class Stmt;

namespace ento {

class BasicValueFactory;
class StoreManager;

///A symbol representing the value stored at a MemRegion.
class SymbolRegionValue : public SymbolData {
  const TypedValueRegion *R;

  friend class SymExprAllocator;
  SymbolRegionValue(SymbolID sym, const TypedValueRegion *r)
      : SymbolData(ClassKind, sym), R(r) {
    assert(r);
    assert(isValidTypeForSymbol(r->getValueType()));
  }

public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const TypedValueRegion *getRegion() const { return R; }

  static void Profile(llvm::FoldingSetNodeID& profile, const TypedValueRegion* R) {
    profile.AddInteger((unsigned)ClassKind);
    profile.AddPointer(R);
  }

  void Profile(llvm::FoldingSetNodeID& profile) override {
    Profile(profile, R);
  }

  StringRef getKindStr() const override;

  void dumpToStream(raw_ostream &os) const override;
  const MemRegion *getOriginRegion() const override { return getRegion(); }

  QualType getType() const override;

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolRegionValueKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// A symbol representing the result of an expression in the case when we do
/// not know anything about what the expression is.
class SymbolConjured : public SymbolData {
  ConstCFGElementRef Elem;
  QualType T;
  unsigned Count;
  const LocationContext *LCtx;
  const void *SymbolTag;

  friend class SymExprAllocator;
  SymbolConjured(SymbolID sym, ConstCFGElementRef elem,
                 const LocationContext *lctx, QualType t, unsigned count,
                 const void *symbolTag)
      : SymbolData(ClassKind, sym), Elem(elem), T(t), Count(count), LCtx(lctx),
        SymbolTag(symbolTag) {
    assert(lctx);
    assert(isValidTypeForSymbol(t));
  }

public:
  ConstCFGElementRef getCFGElementRef() const { return Elem; }

  // It might return null.
  const Stmt *getStmt() const;

  unsigned getCount() const { return Count; }
  /// It might return null.
  const void *getTag() const { return SymbolTag; }

  QualType getType() const override;

  StringRef getKindStr() const override;

  void dumpToStream(raw_ostream &os) const override;

  static void Profile(llvm::FoldingSetNodeID &profile, ConstCFGElementRef Elem,
                      const LocationContext *LCtx, QualType T, unsigned Count,
                      const void *SymbolTag) {
    profile.AddInteger((unsigned)ClassKind);
    profile.Add(Elem);
    profile.AddPointer(LCtx);
    profile.Add(T);
    profile.AddInteger(Count);
    profile.AddPointer(SymbolTag);
  }

  void Profile(llvm::FoldingSetNodeID& profile) override {
    Profile(profile, Elem, LCtx, T, Count, SymbolTag);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolConjuredKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// A symbol representing the value of a MemRegion whose parent region has
/// symbolic value.
class SymbolDerived : public SymbolData {
  SymbolRef parentSymbol;
  const TypedValueRegion *R;

  friend class SymExprAllocator;
  SymbolDerived(SymbolID sym, SymbolRef parent, const TypedValueRegion *r)
      : SymbolData(ClassKind, sym), parentSymbol(parent), R(r) {
    assert(parent);
    assert(r);
    assert(isValidTypeForSymbol(r->getValueType()));
  }

public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  SymbolRef getParentSymbol() const { return parentSymbol; }
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const TypedValueRegion *getRegion() const { return R; }

  QualType getType() const override;

  StringRef getKindStr() const override;

  void dumpToStream(raw_ostream &os) const override;
  const MemRegion *getOriginRegion() const override { return getRegion(); }

  static void Profile(llvm::FoldingSetNodeID& profile, SymbolRef parent,
                      const TypedValueRegion *r) {
    profile.AddInteger((unsigned)ClassKind);
    profile.AddPointer(r);
    profile.AddPointer(parent);
  }

  void Profile(llvm::FoldingSetNodeID& profile) override {
    Profile(profile, parentSymbol, R);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolDerivedKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// SymbolExtent - Represents the extent (size in bytes) of a bounded region.
///  Clients should not ask the SymbolManager for a region's extent. Always use
///  SubRegion::getExtent instead -- the value returned may not be a symbol.
class SymbolExtent : public SymbolData {
  const SubRegion *R;

  friend class SymExprAllocator;
  SymbolExtent(SymbolID sym, const SubRegion *r)
      : SymbolData(ClassKind, sym), R(r) {
    assert(r);
  }

public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const SubRegion *getRegion() const { return R; }

  QualType getType() const override;

  StringRef getKindStr() const override;

  void dumpToStream(raw_ostream &os) const override;

  static void Profile(llvm::FoldingSetNodeID& profile, const SubRegion *R) {
    profile.AddInteger((unsigned)ClassKind);
    profile.AddPointer(R);
  }

  void Profile(llvm::FoldingSetNodeID& profile) override {
    Profile(profile, R);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolExtentKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// SymbolMetadata - Represents path-dependent metadata about a specific region.
///  Metadata symbols remain live as long as they are marked as in use before
///  dead-symbol sweeping AND their associated regions are still alive.
///  Intended for use by checkers.
class SymbolMetadata : public SymbolData {
  const MemRegion* R;
  const Stmt *S;
  QualType T;
  const LocationContext *LCtx;
  /// Count can be used to differentiate regions corresponding to
  /// different loop iterations, thus, making the symbol path-dependent.
  unsigned Count;
  const void *Tag;

  friend class SymExprAllocator;
  SymbolMetadata(SymbolID sym, const MemRegion *r, const Stmt *s, QualType t,
                 const LocationContext *LCtx, unsigned count, const void *tag)
      : SymbolData(ClassKind, sym), R(r), S(s), T(t), LCtx(LCtx), Count(count),
        Tag(tag) {
    assert(r);
    assert(s);
    assert(isValidTypeForSymbol(t));
    assert(LCtx);
    assert(tag);
  }

  public:
    LLVM_ATTRIBUTE_RETURNS_NONNULL
    const MemRegion *getRegion() const { return R; }

    LLVM_ATTRIBUTE_RETURNS_NONNULL
    const Stmt *getStmt() const { return S; }

    LLVM_ATTRIBUTE_RETURNS_NONNULL
    const LocationContext *getLocationContext() const { return LCtx; }

    unsigned getCount() const { return Count; }

    LLVM_ATTRIBUTE_RETURNS_NONNULL
    const void *getTag() const { return Tag; }

    QualType getType() const override;

    StringRef getKindStr() const override;

    void dumpToStream(raw_ostream &os) const override;

    static void Profile(llvm::FoldingSetNodeID &profile, const MemRegion *R,
                        const Stmt *S, QualType T, const LocationContext *LCtx,
                        unsigned Count, const void *Tag) {
      profile.AddInteger((unsigned)ClassKind);
      profile.AddPointer(R);
      profile.AddPointer(S);
      profile.Add(T);
      profile.AddPointer(LCtx);
      profile.AddInteger(Count);
      profile.AddPointer(Tag);
    }

  void Profile(llvm::FoldingSetNodeID& profile) override {
    Profile(profile, R, S, T, LCtx, Count, Tag);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolMetadataKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// Represents a cast expression.
class SymbolCast : public SymExpr {
  const SymExpr *Operand;

  /// Type of the operand.
  QualType FromTy;

  /// The type of the result.
  QualType ToTy;

  friend class SymExprAllocator;
  SymbolCast(SymbolID Sym, const SymExpr *In, QualType From, QualType To)
      : SymExpr(ClassKind, Sym), Operand(In), FromTy(From), ToTy(To) {
    assert(In);
    assert(isValidTypeForSymbol(From));
    // FIXME: GenericTaintChecker creates symbols of void type.
    // Otherwise, 'To' should also be a valid type.
  }

public:
  unsigned computeComplexity() const override {
    if (Complexity == 0)
      Complexity = 1 + Operand->computeComplexity();
    return Complexity;
  }

  QualType getType() const override { return ToTy; }

  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const SymExpr *getOperand() const { return Operand; }

  void dumpToStream(raw_ostream &os) const override;

  static void Profile(llvm::FoldingSetNodeID& ID,
                      const SymExpr *In, QualType From, QualType To) {
    ID.AddInteger((unsigned)ClassKind);
    ID.AddPointer(In);
    ID.Add(From);
    ID.Add(To);
  }

  void Profile(llvm::FoldingSetNodeID& ID) override {
    Profile(ID, Operand, FromTy, ToTy);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = SymbolCastKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// Represents a symbolic expression involving a unary operator.
class UnarySymExpr : public SymExpr {
  const SymExpr *Operand;
  UnaryOperator::Opcode Op;
  QualType T;

  friend class SymExprAllocator;
  UnarySymExpr(SymbolID Sym, const SymExpr *In, UnaryOperator::Opcode Op,
               QualType T)
      : SymExpr(ClassKind, Sym), Operand(In), Op(Op), T(T) {
    // Note, some unary operators are modeled as a binary operator. E.g. ++x is
    // modeled as x + 1.
    assert((Op == UO_Minus || Op == UO_Not) && "non-supported unary expression");
    // Unary expressions are results of arithmetic. Pointer arithmetic is not
    // handled by unary expressions, but it is instead handled by applying
    // sub-regions to regions.
    assert(isValidTypeForSymbol(T) && "non-valid type for unary symbol");
    assert(!Loc::isLocType(T) && "unary symbol should be nonloc");
  }

public:
  unsigned computeComplexity() const override {
    if (Complexity == 0)
      Complexity = 1 + Operand->computeComplexity();
    return Complexity;
  }

  const SymExpr *getOperand() const { return Operand; }
  UnaryOperator::Opcode getOpcode() const { return Op; }
  QualType getType() const override { return T; }

  void dumpToStream(raw_ostream &os) const override;

  static void Profile(llvm::FoldingSetNodeID &ID, const SymExpr *In,
                      UnaryOperator::Opcode Op, QualType T) {
    ID.AddInteger((unsigned)ClassKind);
    ID.AddPointer(In);
    ID.AddInteger(Op);
    ID.Add(T);
  }

  void Profile(llvm::FoldingSetNodeID &ID) override {
    Profile(ID, Operand, Op, T);
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = UnarySymExprKind;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// Represents a symbolic expression involving a binary operator
class BinarySymExpr : public SymExpr {
  BinaryOperator::Opcode Op;
  QualType T;

protected:
  BinarySymExpr(SymbolID Sym, Kind k, BinaryOperator::Opcode op, QualType t)
      : SymExpr(k, Sym), Op(op), T(t) {
    assert(classof(this));
    // Binary expressions are results of arithmetic. Pointer arithmetic is not
    // handled by binary expressions, but it is instead handled by applying
    // sub-regions to regions.
    assert(isValidTypeForSymbol(t) && !Loc::isLocType(t));
  }

public:
  // FIXME: We probably need to make this out-of-line to avoid redundant
  // generation of virtual functions.
  QualType getType() const override { return T; }

  BinaryOperator::Opcode getOpcode() const { return Op; }

  // Implement isa<T> support.
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) {
    return K >= BEGIN_BINARYSYMEXPRS && K <= END_BINARYSYMEXPRS;
  }

protected:
  static unsigned computeOperandComplexity(const SymExpr *Value) {
    return Value->computeComplexity();
  }
  static unsigned computeOperandComplexity(const llvm::APSInt &Value) {
    return 1;
  }

  static const llvm::APSInt *getPointer(APSIntPtr Value) { return Value.get(); }
  static const SymExpr *getPointer(const SymExpr *Value) { return Value; }

  static void dumpToStreamImpl(raw_ostream &os, const SymExpr *Value);
  static void dumpToStreamImpl(raw_ostream &os, const llvm::APSInt &Value);
  static void dumpToStreamImpl(raw_ostream &os, BinaryOperator::Opcode op);
};

/// Template implementation for all binary symbolic expressions
template <class LHSTYPE, class RHSTYPE, SymExpr::Kind ClassK>
class BinarySymExprImpl : public BinarySymExpr {
  LHSTYPE LHS;
  RHSTYPE RHS;

  friend class SymExprAllocator;
  BinarySymExprImpl(SymbolID Sym, LHSTYPE lhs, BinaryOperator::Opcode op,
                    RHSTYPE rhs, QualType t)
      : BinarySymExpr(Sym, ClassKind, op, t), LHS(lhs), RHS(rhs) {
    assert(getPointer(lhs));
    assert(getPointer(rhs));
  }

public:
  void dumpToStream(raw_ostream &os) const override {
    dumpToStreamImpl(os, LHS);
    dumpToStreamImpl(os, getOpcode());
    dumpToStreamImpl(os, RHS);
  }

  LHSTYPE getLHS() const { return LHS; }
  RHSTYPE getRHS() const { return RHS; }

  unsigned computeComplexity() const override {
    if (Complexity == 0)
      Complexity =
          computeOperandComplexity(RHS) + computeOperandComplexity(LHS);
    return Complexity;
  }

  static void Profile(llvm::FoldingSetNodeID &ID, LHSTYPE lhs,
                      BinaryOperator::Opcode op, RHSTYPE rhs, QualType t) {
    ID.AddInteger((unsigned)ClassKind);
    ID.AddPointer(getPointer(lhs));
    ID.AddInteger(op);
    ID.AddPointer(getPointer(rhs));
    ID.Add(t);
  }

  void Profile(llvm::FoldingSetNodeID &ID) override {
    Profile(ID, LHS, getOpcode(), RHS, getType());
  }

  // Implement isa<T> support.
  static constexpr Kind ClassKind = ClassK;
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) { return K == ClassKind; }
};

/// Represents a symbolic expression like 'x' + 3.
using SymIntExpr = BinarySymExprImpl<const SymExpr *, APSIntPtr,
                                     SymExpr::Kind::SymIntExprKind>;

/// Represents a symbolic expression like 3 - 'x'.
using IntSymExpr = BinarySymExprImpl<APSIntPtr, const SymExpr *,
                                     SymExpr::Kind::IntSymExprKind>;

/// Represents a symbolic expression like 'x' + 'y'.
using SymSymExpr = BinarySymExprImpl<const SymExpr *, const SymExpr *,
                                     SymExpr::Kind::SymSymExprKind>;

class SymExprAllocator {
  SymbolID NextSymbolID = 0;
  llvm::BumpPtrAllocator &Alloc;

public:
  explicit SymExprAllocator(llvm::BumpPtrAllocator &Alloc) : Alloc(Alloc) {}

  template <class SymT, typename... ArgsT> SymT *make(ArgsT &&...Args) {
    return new (Alloc) SymT(nextID(), std::forward<ArgsT>(Args)...);
  }

private:
  SymbolID nextID() { return NextSymbolID++; }
};

class SymbolManager {
  using DataSetTy = llvm::FoldingSet<SymExpr>;
  using SymbolDependTy =
      llvm::DenseMap<SymbolRef, std::unique_ptr<SymbolRefSmallVectorTy>>;

  DataSetTy DataSet;

  /// Stores the extra dependencies between symbols: the data should be kept
  /// alive as long as the key is live.
  SymbolDependTy SymbolDependencies;

  SymExprAllocator Alloc;
  BasicValueFactory &BV;
  ASTContext &Ctx;

public:
  SymbolManager(ASTContext &ctx, BasicValueFactory &bv,
                llvm::BumpPtrAllocator &bpalloc)
      : SymbolDependencies(16), Alloc(bpalloc), BV(bv), Ctx(ctx) {}

  static bool canSymbolicate(QualType T);

  /// Create or retrieve a SymExpr of type \p SymExprT for the given arguments.
  /// Use the arguments to check for an existing SymExpr and return it,
  /// otherwise, create a new one and keep a pointer to it to avoid duplicates.
  template <typename SymExprT, typename... Args>
  const SymExprT *acquire(Args &&...args);

  const SymbolConjured *conjureSymbol(ConstCFGElementRef Elem,
                                      const LocationContext *LCtx, QualType T,
                                      unsigned VisitCount,
                                      const void *SymbolTag = nullptr) {

    return acquire<SymbolConjured>(Elem, LCtx, T, VisitCount, SymbolTag);
  }

  QualType getType(const SymExpr *SE) const {
    return SE->getType();
  }

  /// Add artificial symbol dependency.
  ///
  /// The dependent symbol should stay alive as long as the primary is alive.
  void addSymbolDependency(const SymbolRef Primary, const SymbolRef Dependent);

  const SymbolRefSmallVectorTy *getDependentSymbols(const SymbolRef Primary);

  ASTContext &getContext() { return Ctx; }
  BasicValueFactory &getBasicVals() { return BV; }
};

/// A class responsible for cleaning up unused symbols.
class SymbolReaper {
  enum SymbolStatus {
    NotProcessed,
    HaveMarkedDependents
  };

  using SymbolSetTy = llvm::DenseSet<SymbolRef>;
  using SymbolMapTy = llvm::DenseMap<SymbolRef, SymbolStatus>;
  using RegionSetTy = llvm::DenseSet<const MemRegion *>;

  SymbolMapTy TheLiving;
  SymbolSetTy MetadataInUse;

  RegionSetTy LiveRegionRoots;
  // The lazily copied regions are locations for which a program
  // can access the value stored at that location, but not its address.
  // These regions are constructed as a set of regions referred to by
  // lazyCompoundVal.
  RegionSetTy LazilyCopiedRegionRoots;

  const StackFrameContext *LCtx;
  const Stmt *Loc;
  SymbolManager& SymMgr;
  StoreRef reapedStore;
  llvm::DenseMap<const MemRegion *, unsigned> includedRegionCache;

public:
  /// Construct a reaper object, which removes everything which is not
  /// live before we execute statement s in the given location context.
  ///
  /// If the statement is NULL, everything is this and parent contexts is
  /// considered live.
  /// If the stack frame context is NULL, everything on stack is considered
  /// dead.
  SymbolReaper(const StackFrameContext *Ctx, const Stmt *s,
               SymbolManager &symmgr, StoreManager &storeMgr)
      : LCtx(Ctx), Loc(s), SymMgr(symmgr), reapedStore(nullptr, storeMgr) {}

  /// It might return null.
  const LocationContext *getLocationContext() const { return LCtx; }

  bool isLive(SymbolRef sym);
  bool isLiveRegion(const MemRegion *region);
  bool isLive(const Expr *ExprVal, const LocationContext *LCtx) const;
  bool isLive(const VarRegion *VR, bool includeStoreBindings = false) const;

  /// Unconditionally marks a symbol as live.
  ///
  /// This should never be
  /// used by checkers, only by the state infrastructure such as the store and
  /// environment. Checkers should instead use metadata symbols and markInUse.
  void markLive(SymbolRef sym);

  /// Marks a symbol as important to a checker.
  ///
  /// For metadata symbols,
  /// this will keep the symbol alive as long as its associated region is also
  /// live. For other symbols, this has no effect; checkers are not permitted
  /// to influence the life of other symbols. This should be used before any
  /// symbol marking has occurred, i.e. in the MarkLiveSymbols callback.
  void markInUse(SymbolRef sym);

  llvm::iterator_range<RegionSetTy::const_iterator> regions() const {
    return LiveRegionRoots;
  }

  /// Returns whether or not a symbol has been confirmed dead.
  ///
  /// This should only be called once all marking of dead symbols has completed.
  /// (For checkers, this means only in the checkDeadSymbols callback.)
  bool isDead(SymbolRef sym) {
    return !isLive(sym);
  }

  void markLive(const MemRegion *region);
  void markLazilyCopied(const MemRegion *region);
  void markElementIndicesLive(const MemRegion *region);

  /// Set to the value of the symbolic store after
  /// StoreManager::removeDeadBindings has been called.
  void setReapedStore(StoreRef st) { reapedStore = st; }

private:
  bool isLazilyCopiedRegion(const MemRegion *region) const;
  // A readable region is a region that live or lazily copied.
  // Any symbols that refer to values in regions are alive if the region
  // is readable.
  bool isReadableRegion(const MemRegion *region);

  /// Mark the symbols dependent on the input symbol as live.
  void markDependentsLive(SymbolRef sym);
};

class SymbolVisitor {
protected:
  ~SymbolVisitor() = default;

public:
  SymbolVisitor() = default;
  SymbolVisitor(const SymbolVisitor &) = default;
  SymbolVisitor(SymbolVisitor &&) {}

  // The copy and move assignment operator is defined as deleted pending further
  // motivation.
  SymbolVisitor &operator=(const SymbolVisitor &) = delete;
  SymbolVisitor &operator=(SymbolVisitor &&) = delete;

  /// A visitor method invoked by ProgramStateManager::scanReachableSymbols.
  ///
  /// The method returns \c true if symbols should continue be scanned and \c
  /// false otherwise.
  virtual bool VisitSymbol(SymbolRef sym) = 0;
  virtual bool VisitMemRegion(const MemRegion *) { return true; }
};

template <typename T, typename... Args>
const T *SymbolManager::acquire(Args &&...args) {
  llvm::FoldingSetNodeID profile;
  T::Profile(profile, args...);
  void *InsertPos;
  SymExpr *SD = DataSet.FindNodeOrInsertPos(profile, InsertPos);
  if (!SD) {
    SD = Alloc.make<T>(std::forward<Args>(args)...);
    DataSet.InsertNode(SD, InsertPos);
  }
  return cast<T>(SD);
}

} // namespace ento

} // namespace clang

// Override the default definition that would use pointer values of SymbolRefs
// to order them, which is unstable due to ASLR.
// Use the SymbolID instead which reflect the order in which the symbols were
// allocated. This is usually stable across runs leading to the stability of
// ConstraintMap and other containers using SymbolRef as keys.
template <>
struct llvm::ImutContainerInfo<clang::ento::SymbolRef>
    : public ImutProfileInfo<clang::ento::SymbolRef> {
  using value_type = clang::ento::SymbolRef;
  using value_type_ref = clang::ento::SymbolRef;
  using key_type = value_type;
  using key_type_ref = value_type_ref;
  using data_type = bool;
  using data_type_ref = bool;

  static key_type_ref KeyOfValue(value_type_ref D) { return D; }
  static data_type_ref DataOfValue(value_type_ref) { return true; }

  static bool isEqual(clang::ento::SymbolRef LHS, clang::ento::SymbolRef RHS) {
    return LHS->getSymbolID() == RHS->getSymbolID();
  }

  static bool isLess(clang::ento::SymbolRef LHS, clang::ento::SymbolRef RHS) {
    return LHS->getSymbolID() < RHS->getSymbolID();
  }

  // This might seem redundant, but it is required because of the way
  // ImmutableSet is implemented through AVLTree:
  // same as ImmutableMap, but with a non-informative "data".
  static bool isDataEqual(data_type_ref, data_type_ref) { return true; }
};

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMBOLMANAGER_H
