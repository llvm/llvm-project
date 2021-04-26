//===- NodeIntrospection.h -----------------------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file contains the implementation of the NodeIntrospection.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/NodeIntrospection.h"

#include "clang/AST/AST.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {

namespace tooling {

void LocationCallFormatterCpp::print(const LocationCall &Call,
                                     llvm::raw_ostream &OS) {
  if (const LocationCall *On = Call.on()) {
    print(*On, OS);
    if (On->returnsPointer())
      OS << "->";
    else
      OS << '.';
  }

  OS << Call.name() << "()";
}

std::string LocationCallFormatterCpp::format(const LocationCall &Call) {
  std::string Result;
  llvm::raw_string_ostream OS(Result);
  print(Call, OS);
  OS.flush();
  return Result;
}

namespace internal {

static bool locationCallLessThan(const LocationCall *LHS,
                                 const LocationCall *RHS) {
  if (!LHS && !RHS)
    return false;
  if (LHS && !RHS)
    return true;
  if (!LHS && RHS)
    return false;
  auto compareResult = LHS->name().compare(RHS->name());
  if (compareResult < 0)
    return true;
  if (compareResult > 0)
    return false;
  return locationCallLessThan(LHS->on(), RHS->on());
}

bool RangeLessThan::operator()(
    std::pair<SourceRange, SharedLocationCall> const &LHS,
    std::pair<SourceRange, SharedLocationCall> const &RHS) const {
  if (LHS.first.getBegin() < RHS.first.getBegin())
    return true;
  else if (LHS.first.getBegin() != RHS.first.getBegin())
    return false;

  if (LHS.first.getEnd() < RHS.first.getEnd())
    return true;
  else if (LHS.first.getEnd() != RHS.first.getEnd())
    return false;

  return locationCallLessThan(LHS.second.get(), RHS.second.get());
}
bool RangeLessThan::operator()(
    std::pair<SourceLocation, SharedLocationCall> const &LHS,
    std::pair<SourceLocation, SharedLocationCall> const &RHS) const {
  if (LHS.first == RHS.first)
    return locationCallLessThan(LHS.second.get(), RHS.second.get());
  return LHS.first < RHS.first;
}
} // namespace internal

} // namespace tooling
} // namespace clang

#include "clang/Tooling/NodeIntrospection.inc"
