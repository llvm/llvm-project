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

Context Context::empty() { return Context(/*Data=*/nullptr); }

Context::Context(std::shared_ptr<const Data> DataPtr)
    : DataPtr(std::move(DataPtr)) {}

Context Context::clone() const { return Context(DataPtr); }

} // namespace clangd
} // namespace clang
