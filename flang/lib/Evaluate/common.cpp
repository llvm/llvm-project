//===-- lib/Evaluate/common.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/common.h"
#include "flang/Common/idioms.h"

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

void RealFlagWarnings(
    FoldingContext &context, const RealFlags &flags, const char *operation) {
  static constexpr auto warning{common::UsageWarning::FoldingException};
  if (flags.test(RealFlag::Overflow)) {
    context.Warn(warning, "overflow on %s"_warn_en_US, operation);
  }
  if (flags.test(RealFlag::DivideByZero)) {
    if (std::strcmp(operation, "division") == 0) {
      context.Warn(warning, "division by zero"_warn_en_US);
    } else {
      context.Warn(warning, "division by zero on %s"_warn_en_US, operation);
    }
  }
  if (flags.test(RealFlag::InvalidArgument)) {
    context.Warn(warning, "invalid argument on %s"_warn_en_US, operation);
  }
  if (flags.test(RealFlag::Underflow)) {
    context.Warn(warning, "underflow on %s"_warn_en_US, operation);
  }
}

ConstantSubscript &FoldingContext::StartImpliedDo(
    parser::CharBlock name, ConstantSubscript n) {
  auto pair{impliedDos_.insert(std::make_pair(name, n))};
  CHECK(pair.second);
  return pair.first->second;
}

std::optional<ConstantSubscript> FoldingContext::GetImpliedDo(
    parser::CharBlock name) const {
  if (auto iter{impliedDos_.find(name)}; iter != impliedDos_.cend()) {
    return {iter->second};
  } else {
    return std::nullopt;
  }
}

void FoldingContext::EndImpliedDo(parser::CharBlock name) {
  auto iter{impliedDos_.find(name)};
  if (iter != impliedDos_.end()) {
    impliedDos_.erase(iter);
  }
}
} // namespace Fortran::evaluate
