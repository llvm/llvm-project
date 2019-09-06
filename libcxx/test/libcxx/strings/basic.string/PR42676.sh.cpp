//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Regression test for PR42676.

// RUN: %cxx %flags %s -o %t.exe %compile_flags %link_flags -D_LIBCPP_HIDE_FROM_ABI_PER_TU
// RUN: %run

#include <memory>
#include <string>

int main() {
    std::string s1(10u, '-', std::allocator<char>()); (void)s1;
    std::string s2("hello", std::allocator<char>()); (void)s2;
    std::string s3("hello"); (void)s3;
}
