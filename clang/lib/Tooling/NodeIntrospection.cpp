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

namespace clang {

namespace tooling {

std::string LocationCallFormatterCpp::format(LocationCall *Call) {
  SmallVector<LocationCall *> vec;
  while (Call) {
    vec.push_back(Call);
    Call = Call->on();
  }
  std::string result;
  for (auto *VecCall : llvm::reverse(llvm::makeArrayRef(vec).drop_front())) {
    result +=
        (VecCall->name() + "()" + (VecCall->returnsPointer() ? "->" : "."))
            .str();
  }
  result += (vec.back()->name() + "()").str();
  return result;
}

namespace internal {
bool RangeLessThan::operator()(
    std::pair<SourceRange, std::shared_ptr<LocationCall>> const &LHS,
    std::pair<SourceRange, std::shared_ptr<LocationCall>> const &RHS) const {
  if (!LHS.first.isValid() || !RHS.first.isValid())
    return false;

  if (LHS.first.getBegin() < RHS.first.getBegin())
    return true;
  else if (LHS.first.getBegin() != RHS.first.getBegin())
    return false;

  if (LHS.first.getEnd() < RHS.first.getEnd())
    return true;
  else if (LHS.first.getEnd() != RHS.first.getEnd())
    return false;

  return LHS.second->name() < RHS.second->name();
}
} // namespace internal

} // namespace tooling
} // namespace clang

#include "clang/Tooling/NodeIntrospection.inc"
