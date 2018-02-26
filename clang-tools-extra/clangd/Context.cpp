//===--- Context.cpp -----------------------------------------*- C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//

#include "Context.h"
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
