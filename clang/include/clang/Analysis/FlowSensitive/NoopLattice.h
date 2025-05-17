//===-- NoopLattice.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the lattice with exactly one element.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOP_LATTICE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOP_LATTICE_H

#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include <memory>
#include <ostream>

namespace clang {
namespace dataflow {

/// Trivial lattice for dataflow analysis with exactly one element.
///
/// Useful for analyses that only need the Environment and nothing more.
class NoopLattice : public llvm::RTTIExtends<NoopLattice, DataflowLattice> {
public:
  /// For RTTI.
  inline static char ID = 0;

  bool isEqual(const DataflowLattice &Other) const override {
    return llvm::isa<NoopLattice>(Other);
  }

  std::unique_ptr<DataflowLattice> clone() override {
    return std::make_unique<NoopLattice>();
  }

  LatticeEffect join(const DataflowLattice &Other) override {
    assert(llvm::isa<NoopLattice>(Other));
    return LatticeJoinEffect::Unchanged;
  }

  bool operator==(const NoopLattice &Other) const { return true; }
};

inline std::ostream &operator<<(std::ostream &OS, const NoopLattice &) {
  return OS << "noop";
}

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOP_LATTICE_H
