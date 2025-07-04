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

// RUN: %{cxx} %s %{flags} %{compile_flags} -O3 -c -S -emit-llvm -o - | %{check-output}

#include <algorithm>

int* test1(int* first, int* last, int* out) {
  // CHECK:      define
  // CHECK-SAME: test1
  // CHECK:      tail call void @llvm.memmove
  return std::copy(first, last, out);
}
