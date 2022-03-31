//===-- SourceLocationsLattice.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a lattice that collects source locations of interest.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOURCELOCATIONS_LATTICE_H
#define LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOURCELOCATIONS_LATTICE_H

#include "clang/AST/ASTContext.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseSet.h"
#include <string>
#include <utility>

namespace clang {
namespace dataflow {

/// Lattice for dataflow analysis that keeps track of a set of source locations.
///
/// Bottom is the empty set, join is set union, and equality is set equality.
///
/// FIXME: Generalize into a (templated) PowerSetLattice.
class SourceLocationsLattice {
public:
  SourceLocationsLattice() = default;

  explicit SourceLocationsLattice(llvm::DenseSet<SourceLocation> Locs)
      : Locs(std::move(Locs)) {}

  bool operator==(const SourceLocationsLattice &Other) const {
    return Locs == Other.Locs;
  }

  bool operator!=(const SourceLocationsLattice &Other) const {
    return !(*this == Other);
  }

  LatticeJoinEffect join(const SourceLocationsLattice &Other);

  llvm::DenseSet<SourceLocation> &getSourceLocations() { return Locs; }

  const llvm::DenseSet<SourceLocation> &getSourceLocations() const {
    return Locs;
  }

private:
  llvm::DenseSet<SourceLocation> Locs;
};

/// Returns a string that represents the source locations of the lattice.
std::string DebugString(const SourceLocationsLattice &Lattice,
                        const ASTContext &Context);

} // namespace dataflow
} // namespace clang

#endif // LLVM_CLANG_ANALYSIS_FLOWSENSITIVE_SOURCELOCATIONS_LATTICE_H
