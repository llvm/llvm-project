//== SMTSort.h --------------------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic Sort API, which will be the base class
//  for every SMT solver sort specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSORT_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTSORT_H

namespace clang {
namespace ento {

class SMTSort {
public:
  SMTSort() = default;
  virtual ~SMTSort() = default;

  virtual bool isBitvectorSort() const { return isBitvectorSortImpl(); }
  virtual bool isFloatSort() const { return isFloatSortImpl(); }
  virtual bool isBooleanSort() const { return isBooleanSortImpl(); }

  virtual unsigned getBitvectorSortSize() const {
    assert(isBitvectorSort() && "Not a bitvector sort!");
    unsigned Size = getBitvectorSortSizeImpl();
    assert(Size && "Size is zero!");
    return Size;
  };

  virtual unsigned getFloatSortSize() const {
    assert(isFloatSort() && "Not a floating-point sort!");
    unsigned Size = getFloatSortSizeImpl();
    assert(Size && "Size is zero!");
    return Size;
  };

  friend bool operator==(SMTSort const &LHS, SMTSort const &RHS) {
    return LHS.equal_to(RHS);
  }

  virtual void print(raw_ostream &OS) const = 0;

  LLVM_DUMP_METHOD void dump() const { print(llvm::errs()); }

protected:
  virtual bool equal_to(SMTSort const &other) const = 0;

  virtual bool isBitvectorSortImpl() const = 0;

  virtual bool isFloatSortImpl() const = 0;

  virtual bool isBooleanSortImpl() const = 0;

  virtual unsigned getBitvectorSortSizeImpl() const = 0;

  virtual unsigned getFloatSortSizeImpl() const = 0;
};

using SMTSortRef = std::shared_ptr<SMTSort>;

} // namespace ento
} // namespace clang

#endif
