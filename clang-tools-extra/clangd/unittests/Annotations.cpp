//===--- Annotations.cpp - Annotated source code for unit tests --*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Annotations.h"
#include "SourceCode.h"

namespace clang {
namespace clangd {

Position Annotations::point(llvm::StringRef Name) const {
  return offsetToPosition(code(), Base::point(Name));
}

std::vector<Position> Annotations::points(llvm::StringRef Name) const {
  auto Offsets = Base::points(Name);

  std::vector<Position> Ps;
  Ps.reserve(Offsets.size());
  for (size_t O : Offsets)
    Ps.push_back(offsetToPosition(code(), O));

  return Ps;
}

static clangd::Range toLSPRange(llvm::StringRef Code, Annotations::Range R) {
  clangd::Range LSPRange;
  LSPRange.start = offsetToPosition(Code, R.Begin);
  LSPRange.end = offsetToPosition(Code, R.End);
  return LSPRange;
}

clangd::Range Annotations::range(llvm::StringRef Name) const {
  return toLSPRange(code(), Base::range(Name));
}

std::vector<clangd::Range> Annotations::ranges(llvm::StringRef Name) const {
  auto OffsetRanges = Base::ranges(Name);

  std::vector<clangd::Range> Rs;
  Rs.reserve(OffsetRanges.size());
  for (Annotations::Range R : OffsetRanges)
    Rs.push_back(toLSPRange(code(), R));

  return Rs;
}

} // namespace clangd
} // namespace clang
