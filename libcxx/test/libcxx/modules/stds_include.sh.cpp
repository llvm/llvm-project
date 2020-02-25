//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Test that we can include libc++ headers when building with modules
// enabled in various Standard modes. This is a common source of breakage
// since the 'std' module will include all headers, so if something in a
// header fails under a standard mode, importing anything will fail.

// This test fails on Windows because the underlying libc headers on Windows
// are not modular
// XFAIL: LIBCXX-WINDOWS-FIXME

// REQUIRES: modules-support

// NOTE: The -std=XXX flag is present in %flags, so we overwrite it by passing it after %flags.
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++98 %s
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++03 %s
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++11 %s
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++14 %s
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++17 %s
// RUN: %cxx %flags %compile_flags -fmodules -fcxx-modules -fsyntax-only -std=c++2a %s

#include <vector>

int main(int, char**) {
  return 0;
}
