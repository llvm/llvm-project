//== SymExpr.h - Management of Symbolic Values ------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines SymExpr and SymbolData.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMEXPR_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SYMEXPR_H

#include "clang/AST/Type.h"
#include "llvm/ADT/FoldingSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace ento {

class MemRegion;

/// \brief Symbolic value. These values used to capture symbolic execution of
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
  Kind K;

protected:
  SymExpr(Kind k) : K(k) {}

  static bool isValidTypeForSymbol(QualType T) {
    // FIXME: Depending on whether we choose to deprecate structural symbols,
    // this may become much stricter.
    return !T.isNull() && !T->isVoidType();
  }

public:
  virtual ~SymExpr() {}

  Kind getKind() const { return K; }

  virtual void dump() const;

  virtual void dumpToStream(raw_ostream &os) const {}

  virtual QualType getType() const = 0;
  virtual void Profile(llvm::FoldingSetNodeID &profile) = 0;

  /// \brief Iterator over symbols that the current symbol depends on.
  ///
  /// For SymbolData, it's the symbol itself; for expressions, it's the
  /// expression symbol and all the operands in it. Note, SymbolDerived is
  /// treated as SymbolData - the iterator will NOT visit the parent region.
  class symbol_iterator {
    SmallVector<const SymExpr *, 5> itr;
    void expand();

  public:
    symbol_iterator() {}
    symbol_iterator(const SymExpr *SE);

    symbol_iterator &operator++();
    const SymExpr *operator*();

    bool operator==(const symbol_iterator &X) const;
    bool operator!=(const symbol_iterator &X) const;
  };

  symbol_iterator symbol_begin() const { return symbol_iterator(this); }
  static symbol_iterator symbol_end() { return symbol_iterator(); }

  unsigned computeComplexity() const;

  /// \brief Find the region from which this symbol originates.
  ///
  /// Whenever the symbol was constructed to denote an unknown value of
  /// a certain memory region, return this region. This method
  /// allows checkers to make decisions depending on the origin of the symbol.
  /// Symbol classes for which the origin region is known include
  /// SymbolRegionValue which denotes the value of the region before
  /// the beginning of the analysis, and SymbolDerived which denotes the value
  /// of a certain memory region after its super region (a memory space or
  /// a larger record region) is default-bound with a certain symbol.
  virtual const MemRegion *getOriginRegion() const { return nullptr; }
};

inline raw_ostream &operator<<(raw_ostream &os,
                               const clang::ento::SymExpr *SE) {
  SE->dumpToStream(os);
  return os;
}

typedef const SymExpr *SymbolRef;
typedef SmallVector<SymbolRef, 2> SymbolRefSmallVectorTy;

typedef unsigned SymbolID;
/// \brief A symbol representing data which can be stored in a memory location
/// (region).
class SymbolData : public SymExpr {
  void anchor() override;
  const SymbolID Sym;

protected:
  SymbolData(Kind k, SymbolID sym) : SymExpr(k), Sym(sym) {
    assert(classof(this));
  }

public:
  ~SymbolData() override {}

  SymbolID getSymbolID() const { return Sym; }

  // Implement isa<T> support.
  static inline bool classof(const SymExpr *SE) {
    Kind k = SE->getKind();
    return k >= BEGIN_SYMBOLS && k <= END_SYMBOLS;
  }
};

} // namespace ento
} // namespace clang

#endif
