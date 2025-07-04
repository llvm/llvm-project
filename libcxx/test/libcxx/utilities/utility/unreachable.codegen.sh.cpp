//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is checking the LLVM IR
// REQUIRES: clang

// UNSUPPORTED: asan, tsan, ubsan, msan

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// RUN: %{cxx} %s %{flags} %{compile_flags} -Wno-missing-noreturn -O3 -c -S -emit-llvm -o - | %{check-output}

// This is an assert instead in debug mode
// UNSUPPORTED: libcpp-hardening-mode=debug

#include <utility>

void test() {
  // CHECK:      define dso_local void
  // CHECK-SAME: test
  // CHECK-NEXT: unreachable
  // CHECK-NEXT: }
  std::unreachable();
}
