//===--- Context.cpp ---------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "support/Context.h"
#include <cassert>

namespace clang {
namespace clangd {

Context Context::empty() { return Context(/*DataPtr=*/nullptr); }

Context::Context(std::shared_ptr<const Data> DataPtr)
    : DataPtr(std::move(DataPtr)) {}

Context Context::clone() const { return Context(DataPtr); }

static Context &currentContext() {
  static thread_local auto C = Context::empty();
  return C;
}

const Context &Context::current() { return currentContext(); }

Context Context::swapCurrent(Context Replacement) {
  std::swap(Replacement, currentContext());
  return Replacement;
}

} // namespace clangd
} // namespace clang
