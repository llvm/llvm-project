//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.

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

