//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::function support for the "blocks" extension

// UNSUPPORTED: c++03

// This test requires the Blocks runtime, which is (only?) available
// on Darwin out-of-the-box.
// REQUIRES: has-fblocks && darwin

// ADDITIONAL_COMPILE_FLAGS: -fblocks

// Test that including <Block.h> before <functional> compiles and runs.
// This is a regression test for an issue where redeclaring the Block runtime
// functions (_Block_copy, _Block_release) inside libc++'s explicit ABI
// annotations incorrectly added an 'abi_tag' attribute to them.

#include <Block.h>
#include <functional>

int main(int, char**) {
  std::function<void()> f = []() {};
  (void)f;
  return 0;
}
