//===- SVals.h - Abstract Values for Static Analysis ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines SVal, Loc, and NonLoc, classes that represent
//  abstract r-values for use with path-sensitive value tracking.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SVALS_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SVALS_H

#include "clang/AST/Expr.h"
#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymExpr.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/ImmutableList.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <cstdint>
#include <utility>

//==------------------------------------------------------------------------==//
//  Base SVal types.
//==------------------------------------------------------------------------==//

namespace clang {

class CXXBaseSpecifier;
class FunctionDecl;
class LabelDecl;

namespace ento {

class CompoundValData;
class LazyCompoundValData;
class MemRegion;
class PointerToMemberData;
class SValBuilder;
class TypedValueRegion;

namespace nonloc {

/// Sub-kinds for NonLoc values.
enum Kind {
#define NONLOC_SVAL(Id, Parent) Id ## Kind,
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.def"
};

} // namespace nonloc

namespace loc {

/// Sub-kinds for Loc values.
enum Kind {
#define LOC_SVAL(Id, Parent) Id ## Kind,
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.def"
};

} // namespace loc

/// SVal - This represents a symbolic expression, which can be either
///  an L-value or an R-value.
///
class SVal {
public:
  enum BaseKind {
    // The enumerators must be representable using 2 bits.
#define BASIC_SVAL(Id, Parent) Id ## Kind,
#define ABSTRACT_SVAL_WITH_KIND(Id, Parent) Id ## Kind,
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.def"
  };
  enum { BaseBits = 2, BaseMask = 0b11 };

protected:
  const void *Data = nullptr;

  /// The lowest 2 bits are a BaseKind (0 -- 3).
  ///  The higher bits are an unsigned "kind" value.
  unsigned Kind = 0;

  explicit SVal(const void *d, bool isLoc, unsigned ValKind)
      : Data(d), Kind((isLoc ? LocKind : NonLocKind) | (ValKind << BaseBits)) {}

  explicit SVal(BaseKind k, const void *D = nullptr) : Data(D), Kind(k) {}

public:
  explicit SVal() = default;

  /// Convert to the specified SVal type, asserting that this SVal is of
  /// the desired type.
  template <typename T> T castAs() const { return llvm::cast<T>(*this); }

  /// Convert to the specified SVal type, returning None if this SVal is
  /// not of the desired type.
  template <typename T> Optional<T> getAs() const {
    return llvm::dyn_cast<T>(*this);
  }

  unsigned getRawKind() const { return Kind; }
  BaseKind getBaseKind() const { return (BaseKind) (Kind & BaseMask); }
  unsigned getSubKind() const { return Kind >> BaseBits; }

  // This method is required for using SVal in a FoldingSetNode.  It
  // extracts a unique signature for this SVal object.
  void Profile(llvm::FoldingSetNodeID &ID) const {
    ID.AddInteger((unsigned) getRawKind());
    ID.AddPointer(Data);
  }

  bool operator==(SVal R) const {
    return getRawKind() == R.getRawKind() && Data == R.Data;
  }

  bool operator!=(SVal R) const { return !(*this == R); }

  bool isUnknown() const {
    return getRawKind() == UnknownValKind;
  }

  bool isUndef() const {
    return getRawKind() == UndefinedValKind;
  }

  bool isUnknownOrUndef() const {
    return getRawKind() <= UnknownValKind;
  }

  bool isValid() const {
    return getRawKind() > UnknownValKind;
  }

  bool isConstant() const;

  bool isConstant(int I) const;

  bool isZeroConstant() const;

  /// getAsFunctionDecl - If this SVal is a MemRegionVal and wraps a
  /// CodeTextRegion wrapping a FunctionDecl, return that FunctionDecl.
  /// Otherwise return 0.
  const FunctionDecl *getAsFunctionDecl() const;

  /// If this SVal is a location and wraps a symbol, return that
  ///  SymbolRef. Otherwise return 0.
  ///
  /// Casts are ignored during lookup.
  /// \param IncludeBaseRegions The boolean that controls whether the search
  /// should continue to the base regions if the region is not symbolic.
  SymbolRef getAsLocSymbol(bool IncludeBaseRegions = false) const;

  /// Get the symbol in the SVal or its base region.
  SymbolRef getLocSymbolInBase() const;

  /// If this SVal wraps a symbol return that SymbolRef.
  /// Otherwise, return 0.
  ///
  /// Casts are ignored during lookup.
  /// \param IncludeBaseRegions The boolean that controls whether the search
  /// should continue to the base regions if the region is not symbolic.
  SymbolRef getAsSymbol(bool IncludeBaseRegions = false) const;

  /// If this SVal is loc::ConcreteInt or nonloc::ConcreteInt,
  /// return a pointer to APSInt which is held in it.
  /// Otherwise, return nullptr.
  const llvm::APSInt *getAsInteger() const;

  const MemRegion *getAsRegion() const;

  /// printJson - Pretty-prints in JSON format.
  void printJson(raw_ostream &Out, bool AddQuotes) const;

  void dumpToStream(raw_ostream &OS) const;
  void dump() const;

  SymExpr::symbol_iterator symbol_begin() const {
    const SymExpr *SE = getAsSymbol(/*IncludeBaseRegions=*/true);
    if (SE)
      return SE->symbol_begin();
    else
      return SymExpr::symbol_iterator();
  }

  SymExpr::symbol_iterator symbol_end() const {
    return SymExpr::symbol_end();
  }

  /// Try to get a reasonable type for the given value.
  ///
  /// \returns The best approximation of the value type or Null.
  /// In theory, all symbolic values should be typed, but this function
  /// is still a WIP and might have a few blind spots.
  ///
  /// \note This function should not be used when the user has access to the
  /// bound expression AST node as well, since AST always has exact types.
  ///
  /// \note Loc values are interpreted as pointer rvalues for the purposes of
  /// this method.
  QualType getType(const ASTContext &) const;
};

inline raw_ostream &operator<<(raw_ostream &os, clang::ento::SVal V) {
  V.dumpToStream(os);
  return os;
}

class UndefinedVal : public SVal {
public:
  UndefinedVal() : SVal(UndefinedValKind) {}
  static bool classof(SVal V) { return V.getBaseKind() == UndefinedValKind; }
};

class DefinedOrUnknownSVal : public SVal {
public:
  // We want calling these methods to be a compiler error since they are
  // tautologically false.
  bool isUndef() const = delete;
  bool isValid() const = delete;

  static bool classof(SVal V) { return !V.isUndef(); }

protected:
  explicit DefinedOrUnknownSVal(const void *d, bool isLoc, unsigned ValKind)
      : SVal(d, isLoc, ValKind) {}
  explicit DefinedOrUnknownSVal(BaseKind k, void *D = nullptr) : SVal(k, D) {}
};

class UnknownVal : public DefinedOrUnknownSVal {
public:
  explicit UnknownVal() : DefinedOrUnknownSVal(UnknownValKind) {}

  static bool classof(SVal V) { return V.getBaseKind() == UnknownValKind; }
};

class DefinedSVal : public DefinedOrUnknownSVal {
public:
  // We want calling these methods to be a compiler error since they are
  // tautologically true/false.
  bool isUnknown() const = delete;
  bool isUnknownOrUndef() const = delete;
  bool isValid() const = delete;

  static bool classof(SVal V) { return !V.isUnknownOrUndef(); }

protected:
  explicit DefinedSVal(const void *d, bool isLoc, unsigned ValKind)
      : DefinedOrUnknownSVal(d, isLoc, ValKind) {}
};

/// Represents an SVal that is guaranteed to not be UnknownVal.
class KnownSVal : public SVal {
public:
  KnownSVal(const DefinedSVal &V) : SVal(V) {}
  KnownSVal(const UndefinedVal &V) : SVal(V) {}
  static bool classof(SVal V) { return !V.isUnknown(); }
};

class NonLoc : public DefinedSVal {
protected:
  explicit NonLoc(unsigned SubKind, const void *d)
      : DefinedSVal(d, false, SubKind) {}

public:
  void dumpToStream(raw_ostream &Out) const;

  static bool isCompoundType(QualType T) {
    return T->isArrayType() || T->isRecordType() ||
           T->isAnyComplexType() || T->isVectorType();
  }

  static bool classof(SVal V) { return V.getBaseKind() == NonLocKind; }
};

class Loc : public DefinedSVal {
protected:
  explicit Loc(unsigned SubKind, const void *D)
      : DefinedSVal(const_cast<void *>(D), true, SubKind) {}

public:
  void dumpToStream(raw_ostream &Out) const;

  static bool isLocType(QualType T) {
    return T->isAnyPointerType() || T->isBlockPointerType() ||
           T->isReferenceType() || T->isNullPtrType();
  }

  static bool classof(SVal V) { return V.getBaseKind() == LocKind; }
};

//==------------------------------------------------------------------------==//
//  Subclasses of NonLoc.
//==------------------------------------------------------------------------==//

namespace nonloc {

/// Represents symbolic expression that isn't a location.
class SymbolVal : public NonLoc {
public:
  SymbolVal() = delete;
  SymbolVal(SymbolRef sym) : NonLoc(SymbolValKind, sym) {
    assert(sym);
    assert(!Loc::isLocType(sym->getType()));
  }

  LLVM_ATTRIBUTE_RETURNS_NONNULL
  SymbolRef getSymbol() const {
    return (const SymExpr *) Data;
  }

  bool isExpression() const {
    return !isa<SymbolData>(getSymbol());
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind && V.getSubKind() == SymbolValKind;
  }

  static bool classof(NonLoc V) { return V.getSubKind() == SymbolValKind; }
};

/// Value representing integer constant.
class ConcreteInt : public NonLoc {
public:
  explicit ConcreteInt(const llvm::APSInt& V) : NonLoc(ConcreteIntKind, &V) {}

  const llvm::APSInt& getValue() const {
    return *static_cast<const llvm::APSInt *>(Data);
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind && V.getSubKind() == ConcreteIntKind;
  }

  static bool classof(NonLoc V) { return V.getSubKind() == ConcreteIntKind; }
};

class LocAsInteger : public NonLoc {
  friend class ento::SValBuilder;

  explicit LocAsInteger(const std::pair<SVal, uintptr_t> &data)
      : NonLoc(LocAsIntegerKind, &data) {
    // We do not need to represent loc::ConcreteInt as LocAsInteger,
    // as it'd collapse into a nonloc::ConcreteInt instead.
    assert(data.first.getBaseKind() == LocKind &&
           (data.first.getSubKind() == loc::MemRegionValKind ||
            data.first.getSubKind() == loc::GotoLabelKind));
  }

public:
  Loc getLoc() const {
    const std::pair<SVal, uintptr_t> *D =
      static_cast<const std::pair<SVal, uintptr_t> *>(Data);
    return D->first.castAs<Loc>();
  }

  unsigned getNumBits() const {
    const std::pair<SVal, uintptr_t> *D =
      static_cast<const std::pair<SVal, uintptr_t> *>(Data);
    return D->second;
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind && V.getSubKind() == LocAsIntegerKind;
  }

  static bool classof(NonLoc V) { return V.getSubKind() == LocAsIntegerKind; }
};

class CompoundVal : public NonLoc {
  friend class ento::SValBuilder;

  explicit CompoundVal(const CompoundValData *D) : NonLoc(CompoundValKind, D) {
    assert(D);
  }

public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const CompoundValData* getValue() const {
    return static_cast<const CompoundValData *>(Data);
  }

  using iterator = llvm::ImmutableList<SVal>::iterator;

  iterator begin() const;
  iterator end() const;

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind && V.getSubKind() == CompoundValKind;
  }

  static bool classof(NonLoc V) { return V.getSubKind() == CompoundValKind; }
};

class LazyCompoundVal : public NonLoc {
  friend class ento::SValBuilder;

  explicit LazyCompoundVal(const LazyCompoundValData *D)
      : NonLoc(LazyCompoundValKind, D) {
    assert(D);
  }

public:
  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const LazyCompoundValData *getCVData() const {
    return static_cast<const LazyCompoundValData *>(Data);
  }

  /// It might return null.
  const void *getStore() const;

  LLVM_ATTRIBUTE_RETURNS_NONNULL
  const TypedValueRegion *getRegion() const;

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind &&
           V.getSubKind() == LazyCompoundValKind;
  }

  static bool classof(NonLoc V) {
    return V.getSubKind() == LazyCompoundValKind;
  }
};

/// Value representing pointer-to-member.
///
/// This value is qualified as NonLoc because neither loading nor storing
/// operations are applied to it. Instead, the analyzer uses the L-value coming
/// from pointer-to-member applied to an object.
/// This SVal is represented by a NamedDecl which can be a member function
/// pointer or a member data pointer and an optional list of CXXBaseSpecifiers.
/// This list is required to accumulate the pointer-to-member cast history to
/// figure out the correct subobject field. In particular, implicit casts grow
/// this list and explicit casts like static_cast shrink this list.
class PointerToMember : public NonLoc {
  friend class ento::SValBuilder;

public:
  using PTMDataType =
      llvm::PointerUnion<const NamedDecl *, const PointerToMemberData *>;

  const PTMDataType getPTMData() const {
    return PTMDataType::getFromOpaqueValue(const_cast<void *>(Data));
  }

  bool isNullMemberPointer() const;

  const NamedDecl *getDecl() const;

  template<typename AdjustedDecl>
  const AdjustedDecl *getDeclAs() const {
    return dyn_cast_or_null<AdjustedDecl>(getDecl());
  }

  using iterator = llvm::ImmutableList<const CXXBaseSpecifier *>::iterator;

  iterator begin() const;
  iterator end() const;

  static bool classof(SVal V) {
    return V.getBaseKind() == NonLocKind &&
           V.getSubKind() == PointerToMemberKind;
  }

  static bool classof(NonLoc V) {
    return V.getSubKind() == PointerToMemberKind;
  }

private:
  explicit PointerToMember(const PTMDataType D)
      : NonLoc(PointerToMemberKind, D.getOpaqueValue()) {}
};

} // namespace nonloc

//==------------------------------------------------------------------------==//
//  Subclasses of Loc.
//==------------------------------------------------------------------------==//

namespace loc {

class GotoLabel : public Loc {
public:
  explicit GotoLabel(const LabelDecl *Label) : Loc(GotoLabelKind, Label) {
    assert(Label);
  }

  const LabelDecl *getLabel() const {
    return static_cast<const LabelDecl *>(Data);
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == LocKind && V.getSubKind() == GotoLabelKind;
  }

  static bool classof(Loc V) { return V.getSubKind() == GotoLabelKind; }
};

class MemRegionVal : public Loc {
public:
  explicit MemRegionVal(const MemRegion* r) : Loc(MemRegionValKind, r) {
    assert(r);
  }

  /// Get the underlining region.
  const MemRegion *getRegion() const {
    return static_cast<const MemRegion *>(Data);
  }

  /// Get the underlining region and strip casts.
  const MemRegion* stripCasts(bool StripBaseCasts = true) const;

  template <typename REGION>
  const REGION* getRegionAs() const {
    return dyn_cast<REGION>(getRegion());
  }

  bool operator==(const MemRegionVal &R) const {
    return getRegion() == R.getRegion();
  }

  bool operator!=(const MemRegionVal &R) const {
    return getRegion() != R.getRegion();
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == LocKind && V.getSubKind() == MemRegionValKind;
  }

  static bool classof(Loc V) { return V.getSubKind() == MemRegionValKind; }
};

class ConcreteInt : public Loc {
public:
  explicit ConcreteInt(const llvm::APSInt& V) : Loc(ConcreteIntKind, &V) {}

  const llvm::APSInt &getValue() const {
    return *static_cast<const llvm::APSInt *>(Data);
  }

  static bool classof(SVal V) {
    return V.getBaseKind() == LocKind && V.getSubKind() == ConcreteIntKind;
  }

  static bool classof(Loc V) { return V.getSubKind() == ConcreteIntKind; }
};

} // namespace loc
} // namespace ento
} // namespace clang

namespace llvm {
template <typename To, typename From>
struct CastInfo<
    To, From,
    std::enable_if_t<std::is_base_of<::clang::ento::SVal, From>::value>>
    : public CastIsPossible<To, ::clang::ento::SVal> {
  using Self = CastInfo<
      To, From,
      std::enable_if_t<std::is_base_of<::clang::ento::SVal, From>::value>>;
  static bool isPossible(const From &V) {
    return To::classof(*static_cast<const ::clang::ento::SVal *>(&V));
  }
  static Optional<To> castFailed() { return Optional<To>{}; }
  static To doCast(const From &f) {
    return *static_cast<const To *>(cast<::clang::ento::SVal>(&f));
  }
  static Optional<To> doCastIfPossible(const From &f) {
    if (!Self::isPossible(f))
      return Self::castFailed();
    return doCast(f);
  }
};
} // namespace llvm

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SVALS_H
