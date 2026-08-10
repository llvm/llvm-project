//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure we have access to the following substitutions at all times:
// - %{cxx}
// - %{flags}
// - %{compile_flags}
// - %{link_flags}
// - %{exec}

// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe

#include <cassert>

int main(int argc, char**) {
  assert(argc == 1);
  return 0;
}
