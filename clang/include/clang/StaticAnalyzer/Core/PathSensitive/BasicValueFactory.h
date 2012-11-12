//=== BasicValueFactory.h - Basic values for Path Sens analysis --*- C++ -*---//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines BasicValueFactory, a class that manages the lifetime
//  of APSInt objects and symbolic constraints used by ExprEngine
//  and related classes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_BASICVALUEFACTORY_H
#define LLVM_CLANG_GR_BASICVALUEFACTORY_H

#include "clang/StaticAnalyzer/Core/PathSensitive/APSIntType.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/StoreRef.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"

namespace clang {
namespace ento {

class CompoundValData : public llvm::FoldingSetNode {
  QualType T;
  llvm::ImmutableList<SVal> L;

public:
  CompoundValData(QualType t, llvm::ImmutableList<SVal> l)
    : T(t), L(l) {}

  typedef llvm::ImmutableList<SVal>::iterator iterator;
  iterator begin() const { return L.begin(); }
  iterator end() const { return L.end(); }

  static void Profile(llvm::FoldingSetNodeID& ID, QualType T,
                      llvm::ImmutableList<SVal> L);

  void Profile(llvm::FoldingSetNodeID& ID) { Profile(ID, T, L); }
};

class LazyCompoundValData : public llvm::FoldingSetNode {
  StoreRef store;
  const TypedValueRegion *region;
public:
  LazyCompoundValData(const StoreRef &st, const TypedValueRegion *r)
    : store(st), region(r) {}

  const void *getStore() const { return store.getStore(); }
  const TypedValueRegion *getRegion() const { return region; }

  static void Profile(llvm::FoldingSetNodeID& ID,
                      const StoreRef &store,
                      const TypedValueRegion *region);

  void Profile(llvm::FoldingSetNodeID& ID) { Profile(ID, store, region); }
};

class BasicValueFactory {
  typedef llvm::FoldingSet<llvm::FoldingSetNodeWrapper<llvm::APSInt> >
          APSIntSetTy;

  ASTContext &Ctx;
  llvm::BumpPtrAllocator& BPAlloc;

  APSIntSetTy   APSIntSet;
  void *        PersistentSVals;
  void *        PersistentSValPairs;

  llvm::ImmutableList<SVal>::Factory SValListFactory;
  llvm::FoldingSet<CompoundValData>  CompoundValDataSet;
  llvm::FoldingSet<LazyCompoundValData> LazyCompoundValDataSet;

  // This is private because external clients should use the factory
  // method that takes a QualType.
  const llvm::APSInt& getValue(uint64_t X, unsigned BitWidth, bool isUnsigned);

public:
  BasicValueFactory(ASTContext &ctx, llvm::BumpPtrAllocator& Alloc)
  : Ctx(ctx), BPAlloc(Alloc), PersistentSVals(0), PersistentSValPairs(0),
    SValListFactory(Alloc) {}

  ~BasicValueFactory();

  ASTContext &getContext() const { return Ctx; }

  const llvm::APSInt& getValue(const llvm::APSInt& X);
  const llvm::APSInt& getValue(const llvm::APInt& X, bool isUnsigned);
  const llvm::APSInt& getValue(uint64_t X, QualType T);

  /// Returns the type of the APSInt used to store values of the given QualType.
  APSIntType getAPSIntType(QualType T) const {
    assert(T->isIntegerType() || Loc::isLocType(T));
    return APSIntType(Ctx.getTypeSize(T),
                      !T->isSignedIntegerOrEnumerationType());
  }

  /// Convert - Create a new persistent APSInt with the same value as 'From'
  ///  but with the bitwidth and signedness of 'To'.
  const llvm::APSInt &Convert(const llvm::APSInt& To,
                              const llvm::APSInt& From) {
    APSIntType TargetType(To);
    if (TargetType == APSIntType(From))
      return From;

    return getValue(TargetType.convert(From));
  }
  
  const llvm::APSInt &Convert(QualType T, const llvm::APSInt &From) {
    APSIntType TargetType = getAPSIntType(T);
    if (TargetType == APSIntType(From))
      return From;
    
    return getValue(TargetType.convert(From));
  }

  const llvm::APSInt& getIntValue(uint64_t X, bool isUnsigned) {
    QualType T = isUnsigned ? Ctx.UnsignedIntTy : Ctx.IntTy;
    return getValue(X, T);
  }

  inline const llvm::APSInt& getMaxValue(const llvm::APSInt &v) {
    return getValue(APSIntType(v).getMaxValue());
  }

  inline const llvm::APSInt& getMinValue(const llvm::APSInt &v) {
    return getValue(APSIntType(v).getMinValue());
  }

  inline const llvm::APSInt& getMaxValue(QualType T) {
    return getValue(getAPSIntType(T).getMaxValue());
  }

  inline const llvm::APSInt& getMinValue(QualType T) {
    return getValue(getAPSIntType(T).getMinValue());
  }

  inline const llvm::APSInt& Add1(const llvm::APSInt& V) {
    llvm::APSInt X = V;
    ++X;
    return getValue(X);
  }

  inline const llvm::APSInt& Sub1(const llvm::APSInt& V) {
    llvm::APSInt X = V;
    --X;
    return getValue(X);
  }

  inline const llvm::APSInt& getZeroWithPtrWidth(bool isUnsigned = true) {
    return getValue(0, Ctx.getTypeSize(Ctx.VoidPtrTy), isUnsigned);
  }

  inline const llvm::APSInt &getIntWithPtrWidth(uint64_t X, bool isUnsigned) {
    return getValue(X, Ctx.getTypeSize(Ctx.VoidPtrTy), isUnsigned);
  }

  inline const llvm::APSInt& getTruthValue(bool b, QualType T) {
    return getValue(b ? 1 : 0, Ctx.getTypeSize(T), false);
  }

  inline const llvm::APSInt& getTruthValue(bool b) {
    return getTruthValue(b, Ctx.getLogicalOperationType());
  }

  const CompoundValData *getCompoundValData(QualType T,
                                            llvm::ImmutableList<SVal> Vals);

  const LazyCompoundValData *getLazyCompoundValData(const StoreRef &store,
                                            const TypedValueRegion *region);

  llvm::ImmutableList<SVal> getEmptySValList() {
    return SValListFactory.getEmptyList();
  }

  llvm::ImmutableList<SVal> consVals(SVal X, llvm::ImmutableList<SVal> L) {
    return SValListFactory.add(X, L);
  }

  const llvm::APSInt* evalAPSInt(BinaryOperator::Opcode Op,
                                     const llvm::APSInt& V1,
                                     const llvm::APSInt& V2);

  const std::pair<SVal, uintptr_t>&
  getPersistentSValWithData(const SVal& V, uintptr_t Data);

  const std::pair<SVal, SVal>&
  getPersistentSValPair(const SVal& V1, const SVal& V2);

  const SVal* getPersistentSVal(SVal X);
};

} // end GR namespace

} // end clang namespace

#endif
