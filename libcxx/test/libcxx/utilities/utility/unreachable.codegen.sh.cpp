//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test is checking the LLVM IR
// REQUIRES: clang

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// RUN: %{cxx} %s %{compile_flags} -Wno-missing-noreturn -O3 -c -S -emit-llvm -o - | %{check-output}

#include <utility>

void test() {
  // CHECK:      define dso_local void
  // CHECK-SAME: test
  // CHECK-NEXT: unreachable
  // CHECK-NEXT: }
  std::unreachable();
}
