//===--- Relation.cpp --------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Relation.h"

#include <algorithm>

namespace clang {
namespace clangd {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const RelationKind R) {
  switch (R) {
  case RelationKind::BaseOf:
    return OS << "BaseOf";
  case RelationKind::OverriddenBy:
    return OS << "OverriddenBy";
  }
  llvm_unreachable("Unhandled RelationKind enum.");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Relation &R) {
  return OS << R.Subject << " " << R.Predicate << " " << R.Object;
}

llvm::iterator_range<RelationSlab::iterator>
RelationSlab::lookup(const SymbolID &Subject, RelationKind Predicate) const {
  auto IterPair = std::equal_range(Relations.begin(), Relations.end(),
                                   Relation{Subject, Predicate, SymbolID{}},
                                   [](const Relation &A, const Relation &B) {
                                     return std::tie(A.Subject, A.Predicate) <
                                            std::tie(B.Subject, B.Predicate);
                                   });
  return {IterPair.first, IterPair.second};
}

RelationSlab RelationSlab::Builder::build() && {
  // Sort in SPO order.
  llvm::sort(Relations);

  // Remove duplicates.
  Relations.erase(std::unique(Relations.begin(), Relations.end()),
                  Relations.end());

  return RelationSlab{std::move(Relations)};
}

} // namespace clangd
} // namespace clang
