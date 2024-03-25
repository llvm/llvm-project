//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <memory>

// This test ensures that we don't use a non-uglified name 'base' in the
// implementation of scoped_allocator_adaptor.
//
// See https://github.com/llvm/llvm-project/issues/78754.

#include <memory>
#include <scoped_allocator>

using ScopedAlloc = std::scoped_allocator_adaptor<std::allocator<int>, std::allocator<int>>;
struct MyBase {
  using base = MyBase;
};
struct MyDerived : ScopedAlloc, MyBase {};

using T = MyDerived::base; // Should be well-formed
