//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <atomic>

// Test that including <atomic> fails to compile when _LIBCPP_HAS_NO_THREADS
// is defined.

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_HAS_NO_THREADS

#include <atomic>

int main(int, char**)
{

  return 0;
}
