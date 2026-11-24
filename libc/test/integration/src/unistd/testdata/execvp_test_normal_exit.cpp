//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Helper test program for execvp integration tests
///
//===----------------------------------------------------------------------===//

#include <signal.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
  char *env = getenv("EXECV_TEST");
  if (env == nullptr)
    raise(SIGUSR1);
  return 0;
}
