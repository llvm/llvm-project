//===--- CIRChecks.h - cir-tidy -----------------------------*- C++ -*-----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CIRTIDY_CHECKS_H
#define LLVM_CLANG_TOOLS_EXTRA_CIRTIDY_CHECKS_H

// FIXME: split LifetimeCheck.cpp into headers and expose the class in a way
// we can directly query the pass name and unique the source of truth.

namespace cir {
namespace checks {
constexpr const char *LifetimeCheckName = "cir-lifetime-check";
}
} // namespace cir

#endif