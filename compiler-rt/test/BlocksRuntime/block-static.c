//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// testfilerunner CONFIG

#include <stdio.h>


int main(int argc, char **argv) {
  static int numberOfSquesals = 5;

  ^{ numberOfSquesals = 6; }();

  if (numberOfSquesals == 6) {
    printf("%s: success\n", argv[0]);
    return 0;
   }
   printf("**** did not update static local, rdar://6177162\n");
   return 1;

}

