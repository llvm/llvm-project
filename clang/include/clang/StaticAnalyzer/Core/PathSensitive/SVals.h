//== SVals.h - Abstract Values for Static Analysis ---------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SVal, Loc, and NonLoc, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_GR_RVALUE_H
#define LLVM_CLANG_GR_RVALUE_H

#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState_Fwd.h"
#include "llvm/ADT/ImmutableList.h"

//==------------------------------------------------------------------------==//
//  Base SVal types.
//==------------------------------------------------------------------------==//

namespace clang {

namespace ento {

class CompoundValData;
class LazyCompoundValData;
class ProgramState;
class BasicValueFactory;
class MemRegion;
class TypedRegion;
class MemRegionManager;
class ProgramStateManager;
class SValBuilder;

/// SVal - This represents a symbolic expression, which can be either
///  an L-value or an R-value.
///
class SVal {
public:
  enum BaseKind {
    // The enumerators must be representable using 2 bits.
    UndefinedKind = 0,  // for subclass UndefinedVal (an uninitialized value)
    UnknownKind = 1,    // for subclass UnknownVal (a void value)
    LocKind = 2,        // for subclass Loc (an L-value)
    NonLocKind = 3      // for subclass NonLoc (an R-value that's not
                        //   an L-value)
  };
  enum { BaseBits = 2, BaseMask = 0x3 };

protected:
  const void *Data;

  /// The lowest 2 bits are a BaseKind (0 -- 3).
  ///  The higher bits are an unsigned "kind" value.
  unsigned Kind;

  explicit SVal(const void *d, bool isLoc, unsigned ValKind)
  : Data(d), Kind((isLoc ? LocKind : NonLocKind) | (ValKind << BaseBits)) {}

  explicit SVal(BaseKind k, const void *D = NULL)
    : Data(D), Kind(k) {}

public:
  explicit SVal() : Data(0), Kind(0) {}

  /// BufferTy - A temporary buffer to hold a set of SVals.
  typedef SmallVector<SVal,5> BufferTy;

  inline unsigned getRawKind() const { return Kind; }
  inline BaseKind getBaseKind() const { return (BaseKind) (Kind & BaseMask); }
  inline unsigned getSubKind() const { return (Kind & ~BaseMask) >> BaseBits; }

  // This method is required for using SVal in a FoldingSetNode.  It
  // extracts a unique signature for this SVal object.
  inline void Profile(llvm::FoldingSetNodeID& ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(Data);
  }

  inline bool operator==(const SVal& R) const {
    return getRawKind() == R.getRawKind() && Data == R.Data;
  }

  inline bool operator!=(const SVal& R) const {
    return !(*this == R);
  }

  inline bool isUnknown() const {
    return getRawKind() == UnknownKind;
  }

  inline bool isUndef() const {
    return getRawKind() == UndefinedKind;
  }

  inline bool isUnknownOrUndef() const {
    return getRawKind() <= UnknownKind;
  }

  inline bool isValid() const {
    return getRawKind() > UnknownKind;
  }

  bool isConstant() const;

  bool isConstant(int I) const;

  bool isZeroConstant() const;

  /// hasConjuredSymbol - If this SVal wraps a conjured symbol, return true;
  bool hasConjuredSymbol() const;

  /// getAsFunctionDecl - If this SVal is a MemRegionVal and wraps a
  /// CodeTextRegion wrapping a FunctionDecl, return that FunctionDecl.
  /// Otherwise return 0.
  const FunctionDecl *getAsFunctionDecl() const;

  /// If this SVal is a location (subclasses Loc) and
  /// wraps a symbol, return that SymbolRef.  Otherwise return 0.
  SymbolRef getAsLocSymbol() const;

  /// Get the symbol in the SVal or its base region.
  SymbolRef getLocSymbolInBase() const;

  /// If this SVal wraps a symbol return that SymbolRef.
  /// Otherwise, return 0.
  SymbolRef getAsSymbol() const;

  /// getAsSymbolicExpression - If this Sval wraps a symbolic expression then
  ///  return that expression.  Otherwise return NULL.
  const SymExpr *getAsSymbolicExpression() const;

  const SymExpr* getAsSymExpr() const;

  const MemRegion *getAsRegion() const;

  void dumpToStream(raw_ostream &OS) const;
  void dump() const;

  SymExpr::symbol_iterator symbol_begin() const {
    const SymExpr *SE = getAsSymbolicExpression();
    if (SE)
      return SE->symbol_begin();
    else
      return SymExpr::symbol_iterator();
  }

  SymExpr::symbol_iterator symbol_end() const { 
    return SymExpr::symbol_end();
  }
};


class UndefinedVal : public SVal {
public:
  UndefinedVal() : SVal(UndefinedKind) {}

  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == UndefinedKind;
  }
};

class DefinedOrUnknownSVal : public SVal {
private:
  // Do not implement.  We want calling these methods to be a compiler
  // error since they are tautologically false.
  bool isUndef() const;
  bool isValid() const;
  
protected:
  explicit DefinedOrUnknownSVal(const void *d, bool isLoc, unsigned ValKind)
    : SVal(d, isLoc, ValKind) {}
  
  explicit DefinedOrUnknownSVal(BaseKind k, void *D = NULL)
    : SVal(k, D) {}
  
public:
    // Implement isa<T> support.
  static inline bool classof(const SVal *V) {
    return !V->isUndef();
  }
};
  
class UnknownVal : public DefinedOrUnknownSVal {
public:
  explicit UnknownVal() : DefinedOrUnknownSVal(UnknownKind) {}
  
  static inline bool classof(const SVal *V) {
    return V->getBaseKind() == UnknownKind;
  }
};

class DefinedSVal : public DefinedOrUnknownSVal {
private:
  // Do not implement.  We want calling these methods to be a compiler
  // error since they are tautologically true/false.
  bool isUnknown() const;
  bool isUnknownOrUndef() const;
  bool isValid() const;  
protected:
  explicit DefinedSVal(const void *d, bool isLoc, unsigned ValKind)
    : DefinedOrUnknownSVal(d, isLoc, ValKind) {}
public:
  // Implement isa<T> support.
  static inline bool classof(const SVal *V) {
    return !V->isUnknownOrUndef();
  }
};

class NonLoc : public DefinedSVal {
protected:
  explicit NonLoc(unsigned SubKind, const void *d)
    : DefinedSVal(d, false, SubKind) {}

public:
  void dumpToStream(raw_ostream &Out) const;

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind;
  }
};

class Loc : public DefinedSVal {
protected:
  explicit Loc(unsigned SubKind, const void *D)
  : DefinedSVal(const_cast<void*>(D), true, SubKind) {}

public:
  void dumpToStream(raw_ostream &Out) const;

  Loc(const Loc& X) : DefinedSVal(X.Data, true, X.getSubKind()) {}

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind;
  }

  static inline bool isLocType(QualType T) {
    return T->isAnyPointerType() || T->isBlockPointerType() || 
           T->isReferenceType();
  }
};

//==------------------------------------------------------------------------==//
//  Subclasses of NonLoc.
//==------------------------------------------------------------------------==//

namespace nonloc {

enum Kind { ConcreteIntKind, SymbolValKind,
            LocAsIntegerKind, CompoundValKind, LazyCompoundValKind };

/// \brief Represents symbolic expression.
class SymbolVal : public NonLoc {
public:
  SymbolVal(SymbolRef sym) : NonLoc(SymbolValKind, sym) {}

  SymbolRef getSymbol() const {
    return (const SymExpr*) Data;
  }

  bool isExpression() {
    return !isa<SymbolData>(getSymbol());
  }

  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == SymbolValKind;
  }

  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == SymbolValKind;
  }
};

/// \brief Value representing integer constant.
class ConcreteInt : public NonLoc {
public:
  explicit ConcreteInt(const llvm::APSInt& V) : NonLoc(ConcreteIntKind, &V) {}

  const llvm::APSInt& getValue() const {
    return *static_cast<const llvm::APSInt*>(Data);
  }

  // Transfer functions for binary/unary operations on ConcreteInts.
  SVal evalBinOp(SValBuilder &svalBuilder, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;

  ConcreteInt evalComplement(SValBuilder &svalBuilder) const;

  ConcreteInt evalMinus(SValBuilder &svalBuilder) const;

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == ConcreteIntKind;
  }

  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};

class LocAsInteger : public NonLoc {
  friend class ento::SValBuilder;

  explicit LocAsInteger(const std::pair<SVal, uintptr_t>& data) :
    NonLoc(LocAsIntegerKind, &data) {
      assert (isa<Loc>(data.first));
    }

public:

  Loc getLoc() const {
    const std::pair<SVal, uintptr_t> *D =
      static_cast<const std::pair<SVal, uintptr_t> *>(Data);
    return cast<Loc>(D->first);
  }

  const Loc& getPersistentLoc() const {
    const std::pair<SVal, uintptr_t> *D =
      static_cast<const std::pair<SVal, uintptr_t> *>(Data);
    const SVal& V = D->first;
    return cast<Loc>(V);
  }

  unsigned getNumBits() const {
    const std::pair<SVal, uintptr_t> *D =
      static_cast<const std::pair<SVal, uintptr_t> *>(Data);
    return D->second;
  }

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == LocAsIntegerKind;
  }

  static inline bool classof(const NonLoc* V) {
    return V->getSubKind() == LocAsIntegerKind;
  }
};

class CompoundVal : public NonLoc {
  friend class ento::SValBuilder;

  explicit CompoundVal(const CompoundValData* D) : NonLoc(CompoundValKind, D) {}

public:
  const CompoundValData* getValue() const {
    return static_cast<const CompoundValData*>(Data);
  }

  typedef llvm::ImmutableList<SVal>::iterator iterator;
  iterator begin() const;
  iterator end() const;

  static bool classof(const SVal* V) {
    return V->getBaseKind() == NonLocKind && V->getSubKind() == CompoundValKind;
  }

  static bool classof(const NonLoc* V) {
    return V->getSubKind() == CompoundValKind;
  }
};

class LazyCompoundVal : public NonLoc {
  friend class ento::SValBuilder;

  explicit LazyCompoundVal(const LazyCompoundValData *D)
    : NonLoc(LazyCompoundValKind, D) {}
public:
  const LazyCompoundValData *getCVData() const {
    return static_cast<const LazyCompoundValData*>(Data);
  }
  const void *getStore() const;
  const TypedRegion *getRegion() const;

  static bool classof(const SVal *V) {
    return V->getBaseKind() == NonLocKind &&
           V->getSubKind() == LazyCompoundValKind;
  }
  static bool classof(const NonLoc *V) {
    return V->getSubKind() == LazyCompoundValKind;
  }
};

} // end namespace ento::nonloc

//==------------------------------------------------------------------------==//
//  Subclasses of Loc.
//==------------------------------------------------------------------------==//

namespace loc {

enum Kind { GotoLabelKind, MemRegionKind, ConcreteIntKind };

class GotoLabel : public Loc {
public:
  explicit GotoLabel(LabelDecl *Label) : Loc(GotoLabelKind, Label) {}

  const LabelDecl *getLabel() const {
    return static_cast<const LabelDecl*>(Data);
  }

  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind && V->getSubKind() == GotoLabelKind;
  }

  static inline bool classof(const Loc* V) {
    return V->getSubKind() == GotoLabelKind;
  }
};


class MemRegionVal : public Loc {
public:
  explicit MemRegionVal(const MemRegion* r) : Loc(MemRegionKind, r) {}

  /// \brief Get the underlining region.
  const MemRegion* getRegion() const {
    return static_cast<const MemRegion*>(Data);
  }

  /// \brief Get the underlining region and strip casts.
  const MemRegion* stripCasts(bool StripBaseCasts = true) const;

  template <typename REGION>
  const REGION* getRegionAs() const {
    return llvm::dyn_cast<REGION>(getRegion());
  }

  inline bool operator==(const MemRegionVal& R) const {
    return getRegion() == R.getRegion();
  }

  inline bool operator!=(const MemRegionVal& R) const {
    return getRegion() != R.getRegion();
  }

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == MemRegionKind;
  }

  static inline bool classof(const Loc* V) {
    return V->getSubKind() == MemRegionKind;
  }
};

class ConcreteInt : public Loc {
public:
  explicit ConcreteInt(const llvm::APSInt& V) : Loc(ConcreteIntKind, &V) {}

  const llvm::APSInt& getValue() const {
    return *static_cast<const llvm::APSInt*>(Data);
  }

  // Transfer functions for binary/unary operations on ConcreteInts.
  SVal evalBinOp(BasicValueFactory& BasicVals, BinaryOperator::Opcode Op,
                 const ConcreteInt& R) const;

  // Implement isa<T> support.
  static inline bool classof(const SVal* V) {
    return V->getBaseKind() == LocKind &&
           V->getSubKind() == ConcreteIntKind;
  }

  static inline bool classof(const Loc* V) {
    return V->getSubKind() == ConcreteIntKind;
  }
};

} // end ento::loc namespace
} // end GR namespace

} // end clang namespace

namespace llvm {
static inline raw_ostream &operator<<(raw_ostream &os,
                                            clang::ento::SVal V) {
  V.dumpToStream(os);
  return os;
}

} // end llvm namespace

#endif
