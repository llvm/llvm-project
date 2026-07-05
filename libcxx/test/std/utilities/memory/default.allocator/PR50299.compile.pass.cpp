//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// Make sure we can use std::allocator<void> in all Standard modes. While the
// explicit specialization for std::allocator<void> was deprecated, using that
// specialization was neither deprecated nor removed (in C++20 it should simply
// start using the primary template).
//
// See https://llvm.org/PR50299.

#include <memory>

std::allocator<void> a;
