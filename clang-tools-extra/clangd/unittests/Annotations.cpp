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
  return pointWithPayload(Name).first;
}

std::pair<Position, llvm::StringRef>
Annotations::pointWithPayload(llvm::StringRef Name) const {
  auto [BasePoint, Payload] = Base::pointWithPayload(Name);
  return {offsetToPosition(code(), BasePoint), Payload};
}

std::vector<Position> Annotations::points(llvm::StringRef Name) const {
  auto BasePoints = Base::points(Name);

  std::vector<Position> Ps;
  Ps.reserve(BasePoints.size());
  for (const auto Point : BasePoints)
    Ps.push_back(offsetToPosition(code(), Point));

  return Ps;
}

std::vector<std::pair<Position, llvm::StringRef>>
Annotations::pointsWithPayload(llvm::StringRef Name) const {
  auto BasePoints = Base::pointsWithPayload(Name);

  std::vector<std::pair<Position, llvm::StringRef>> Ps;
  Ps.reserve(BasePoints.size());
  for (const auto &[Point, Payload] : BasePoints)
    Ps.push_back({offsetToPosition(code(), Point), Payload});

  return Ps;
}

static clangd::Range toLSPRange(llvm::StringRef Code,
                                llvm::Annotations::Range R) {
  clangd::Range LSPRange;
  LSPRange.start = offsetToPosition(Code, R.Begin);
  LSPRange.end = offsetToPosition(Code, R.End);
  return LSPRange;
}

Range Annotations::range(llvm::StringRef Name) const {
  return rangeWithPayload(Name).first;
}

std::pair<clangd::Range, llvm::StringRef>
Annotations::rangeWithPayload(llvm::StringRef Name) const {
  auto [BaseRange, Payload] = Base::rangeWithPayload(Name);
  return {toLSPRange(code(), BaseRange), Payload};
}

std::vector<Range> Annotations::ranges(llvm::StringRef Name) const {
  auto OffsetRanges = Base::ranges(Name);

  std::vector<clangd::Range> Rs;
  Rs.reserve(OffsetRanges.size());
  for (const auto &R : OffsetRanges)
    Rs.push_back(toLSPRange(code(), R));

  return Rs;
}

std::vector<std::pair<clangd::Range, llvm::StringRef>>
Annotations::rangesWithPayload(llvm::StringRef Name) const {
  auto OffsetRanges = Base::rangesWithPayload(Name);

  std::vector<std::pair<clangd::Range, llvm::StringRef>> Rs;
  Rs.reserve(OffsetRanges.size());
  for (const auto &[R, Payload] : OffsetRanges)
    Rs.push_back({toLSPRange(code(), R), Payload});

  return Rs;
}

} // namespace clangd
} // namespace clang
