//===- SymExpr.h - Management of Symbolic Values ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymExpr and SymbolData.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMEXPR_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMEXPR_H

#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Compiler.h"
#include <cassert>

namespace clang {
namespace ento {

class MemRegion;

using SymbolID = unsigned;

/// Symbolic value. These values used to capture symbolic execution of
/// the program.
class SymExpr : public llvm::FoldingSetNode {
  virtual void anchor();

public:
  enum Kind {
#define SYMBOL(Id, Parent) Id##Kind,
#define SYMBOL_RANGE(Id, First, Last) BEGIN_##Id = First, END_##Id = Last,
#include "clang/StaticAnalyzer/Core/PathSensitive/Symbols.def"
  };

private:
  /// A unique identifier for this symbol.
  ///
  /// It is useful for SymbolData to easily differentiate multiple symbols, but
  /// also for "ephemeral" symbols, such as binary operations, because this id
  /// can be used for arranging constraints or equivalence classes instead of
  /// unstable pointer values.
  ///
  /// Note, however, that it can't be used in Profile because SymbolManager
  /// needs to compute Profile before allocating SymExpr.
  const SymbolID Sym;

protected:
  SymExpr(Kind k, SymbolID Sym) : Sym(Sym), K(k) {}

  static bool isValidTypeForSymbol(QualType T) {
    // FIXME: Depending on whether we choose to deprecate structural symbols,
    // this may become much stricter.
    return !T.isNull() && !T->isVoidType();
  }

  mutable unsigned Complexity = 0;

private:
  // Declared last so that subclasses can put small fields into tail padding.
  LLVM_PREFERRED_TYPE(Kind)
  unsigned K : 8;

public:
  virtual ~SymExpr() = default;

  Kind getKind() const { return static_cast<Kind>(K); }

  /// Get a unique identifier for this symbol.
  /// The ID is unique across all SymExprs in a SymbolManager.
  /// They reflect the allocation order of these SymExprs,
  /// and are likely stable across runs.
  /// Used as a key in SymbolRef containers and as part of identity
  /// for SymbolData, e.g. SymbolConjured with ID = 7 is "conj_$7".
  SymbolID getSymbolID() const { return Sym; }

  virtual void dump() const;

  virtual void dumpToStream(raw_ostream &os) const {}

  virtual QualType getType() const = 0;
  virtual void Profile(llvm::FoldingSetNodeID &profile) = 0;

  /// Iterator over symbols that the current symbol depends on.
  ///
  /// For SymbolData, it's the symbol itself; for expressions, it's the
  /// expression symbol and all the operands in it. Note, SymbolDerived is
  /// treated as SymbolData - the iterator will NOT visit the parent region.
  ///
  /// The worklist is owned by the symbol_range, so the iterators are
  /// single-pass and only meant to be compared against the end iterator.
  class symbol_iterator {
    SmallVectorImpl<const SymExpr *> *WL = nullptr;

    void expand(const SymExpr *SE);
    bool atEnd() const { return !WL || WL->empty(); }

  public:
    symbol_iterator() = default;
    explicit symbol_iterator(SmallVectorImpl<const SymExpr *> *WL) : WL(WL) {}

    symbol_iterator &operator++();

    const SymExpr *operator*() const {
      assert(!atEnd() && "attempting to dereference an 'end' iterator");
      return WL->back();
    }

    bool operator==(const symbol_iterator &X) const {
      return atEnd() == X.atEnd();
    }
    bool operator!=(const symbol_iterator &X) const { return !(*this == X); }
  };

  /// A single-pass range of symbol_iterator that owns the worklist.
  class symbol_range {
    SmallVector<const SymExpr *, 5> Worklist;

  public:
    explicit symbol_range(const SymExpr *Root) {
      if (Root)
        Worklist.push_back(Root);
    }
    symbol_range(const symbol_range &) = delete;
    symbol_range &operator=(const symbol_range &) = delete;
    symbol_range(symbol_range &&) = delete;
    symbol_range &operator=(symbol_range &&) = delete;

    symbol_iterator begin() LLVM_LIFETIME_BOUND {
      return symbol_iterator(&Worklist);
    }
    symbol_iterator end() { return symbol_iterator(); }
  };

  symbol_range symbols() const { return symbol_range(this); }

  virtual unsigned computeComplexity() const = 0;

  /// Find the region from which this symbol originates.
  ///
  /// Whenever the symbol was constructed to denote an unknown value of
  /// a certain memory region, return this region. This method
  /// allows checkers to make decisions depending on the origin of the symbol.
  /// Symbol classes for which the origin region is known include
  /// SymbolRegionValue which denotes the value of the region before
  /// the beginning of the analysis, and SymbolDerived which denotes the value
  /// of a certain memory region after its super region (a memory space or
  /// a larger record region) is default-bound with a certain symbol.
  /// It might return null.
  virtual const MemRegion *getOriginRegion() const { return nullptr; }
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const clang::ento::SymExpr *SE) {
  SE->dumpToStream(os);
  return os;
}

using SymbolRef = const SymExpr *;
using SymbolRefSmallVectorTy = SmallVector<SymbolRef, 2>;

/// A symbol representing data which can be stored in a memory location
/// (region).
class SymbolData : public SymExpr {
  void anchor() override;

protected:
  SymbolData(Kind k, SymbolID sym) : SymExpr(k, sym) { assert(classof(this)); }

public:
  ~SymbolData() override = default;

  /// Get a string representation of the kind of the region.
  virtual StringRef getKindStr() const = 0;

  unsigned computeComplexity() const override {
    return 1;
  };

  // Implement isa<T> support.
  static bool classof(const SymExpr *SE) { return classof(SE->getKind()); }
  static constexpr bool classof(Kind K) {
    return K >= BEGIN_SYMBOLS && K <= END_SYMBOLS;
  }
};

inline SymExpr::symbol_iterator &SymExpr::symbol_iterator::operator++() {
  assert(!atEnd() && "attempting to iterate on an 'end' iterator");
  const SymExpr *SE = WL->pop_back_val();
  // SymbolData has no operands, which is by far the most common case.
  if (!isa<SymbolData>(SE))
    expand(SE);
  return *this;
}

} // namespace ento
} // namespace clang

#endif // LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMEXPR_H
