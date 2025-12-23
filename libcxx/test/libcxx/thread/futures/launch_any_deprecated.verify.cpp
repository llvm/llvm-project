//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <future>

// enum class launch;

// Verify that std::launch::any is deprecated.
// It was a draft C++11 feature that was removed, but libc++ kept it as an extension.

#include <future>

void test() {
  (void)std::launch::any; // expected-warning {{std::launch::any is a deprecated extension}}
}
