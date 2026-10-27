//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char **argv) {
  if (argc != 3)
    raise(SIGUSR1);
  if (strcmp(argv[0], "execle_test_normal_exit") != 0)
    raise(SIGUSR1);
  if (strcmp(argv[1], "first") != 0)
    raise(SIGUSR1);
  if (strcmp(argv[2], "second") != 0)
    raise(SIGUSR1);

  char *env = getenv("EXECLE_TEST");
  if (env == nullptr || strcmp(env, "PASS") != 0)
    raise(SIGUSR1);

  return 0;
}
