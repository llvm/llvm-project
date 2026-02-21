//===--- CollectActualArguments.h - flang-tidy ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_TIDY_UTILS_COLLECT_ACTUAL_ARGUMENTS
#define FORTRAN_TIDY_UTILS_COLLECT_ACTUAL_ARGUMENTS

#include "flang/Evaluate/call.h"
#include "flang/Evaluate/traverse.h"

namespace Fortran::evaluate {
using ActualArgumentRef = common::Reference<const ActualArgument>;

inline bool operator<(ActualArgumentRef x, ActualArgumentRef y) {
  return &*x < &*y;
}

using ActualArgumentSet = std::set<evaluate::ActualArgumentRef>;

struct CollectActualArgumentsHelper
    : public evaluate::SetTraverse<CollectActualArgumentsHelper,
                                   ActualArgumentSet> {
  using Base = SetTraverse<CollectActualArgumentsHelper, ActualArgumentSet>;
  CollectActualArgumentsHelper() : Base{*this} {}
  using Base::operator();
  ActualArgumentSet operator()(const evaluate::ActualArgument &arg) const {
    return Combine(ActualArgumentSet{arg},
                   CollectActualArgumentsHelper{}(arg.UnwrapExpr()));
  }
};

template <typename A>
ActualArgumentSet CollectActualArguments(const A &x) {
  return CollectActualArgumentsHelper{}(x);
}

template ActualArgumentSet CollectActualArguments(const semantics::SomeExpr &);

} // namespace Fortran::evaluate

#endif // FORTRAN_TIDY_UTILS_COLLECT_ACTUAL_ARGUMENTS
