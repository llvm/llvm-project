//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Make sure that test-executables can appear in RUN lines and be executed
// effectively. This somewhat difficult-to-understand test checks that when
// we run with a remote executor, test-executables are copied to the remote
// host and their path is fixed up (directly in the command-line) to their
// path on the remote host instead of the local host.
//
// We also check that the path of test-executables is replaced whether they
// appear first in the command-line or not.

// UNSUPPORTED: executor-has-no-bash
// RUN: %{cxx} %s %{flags} %{compile_flags} %{link_flags} -o %t.exe
// RUN: %{exec} %t.exe 0
// RUN: %{exec} bash -c '! %t.exe 1'

#include <cassert>
#include <cstdlib>

int main(int argc, char** argv) {
  assert(argc == 2);
  int ret = std::atoi(argv[1]);
  return ret;
}
