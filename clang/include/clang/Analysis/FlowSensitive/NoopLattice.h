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
#include "clang/Support/Compiler.h"
#include "llvm/ADT/Any.h"
#include <ostream>

namespace clang {
namespace dataflow {

/// Trivial lattice for dataflow analysis with exactly one element.
///
/// Useful for analyses that only need the Environment and nothing more.
class NoopLattice {
public:
  bool operator==(const NoopLattice &Other) const { return true; }

  LatticeJoinEffect join(const NoopLattice &Other) {
    return LatticeJoinEffect::Unchanged;
  }
};

inline std::ostream &operator<<(std::ostream &OS, const NoopLattice &) {
  return OS << "noop";
}

} // namespace dataflow
} // namespace clang

namespace llvm {
// This needs to be exported for ClangAnalysisFlowSensitiveTests so any_cast
// uses the correct address of Any::TypeId from the clang shared library instead
// of creating one in the test executable. when building with
// CLANG_LINK_CLANG_DYLIB
extern template struct CLANG_TEMPLATE_ABI
    Any::TypeId<clang::dataflow::NoopLattice>;
} // namespace llvm

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_NOOP_LATTICE_H
