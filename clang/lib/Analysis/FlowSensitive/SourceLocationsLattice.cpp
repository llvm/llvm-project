//===- SourceLocationsLattice.cpp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements a lattice that collects source locations of interest.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/AST/ASTContext.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <string>
#include <vector>

namespace clang {
namespace dataflow {

LatticeJoinEffect
SourceLocationsLattice::join(const SourceLocationsLattice &Other) {
  auto SizeBefore = Locs.size();
  Locs.insert(Other.Locs.begin(), Other.Locs.end());
  return SizeBefore == Locs.size() ? LatticeJoinEffect::Unchanged
                                   : LatticeJoinEffect::Changed;
}

std::string DebugString(const SourceLocationsLattice &Lattice,
                        const ASTContext &Context) {
  if (Lattice.getSourceLocations().empty())
    return "";

  std::vector<std::string> Locations;
  Locations.reserve(Lattice.getSourceLocations().size());
  for (const clang::SourceLocation &Loc : Lattice.getSourceLocations()) {
    Locations.push_back(Loc.printToString(Context.getSourceManager()));
  }
  std::sort(Locations.begin(), Locations.end());
  std::string result;
  llvm::raw_string_ostream OS(result);
  llvm::interleaveComma(Locations, OS);
  return result;
}

} // namespace dataflow
} // namespace clang
