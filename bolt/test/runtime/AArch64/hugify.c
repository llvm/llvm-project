// Make sure BOLT correctly processes --hugify option

#include <stdio.h>

int g1 = 1;
int g2;
static int sg1 = 1;
static int sg2;

int main(int argc, char **argv) {
  printf("Hello world %p = %d , %p = %d\n", &g1, g1, &sg1, sg1);
  printf("%p = %d , %p = %d\n", &g2, g2, &sg2, sg2);
  return 0;
}

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags -no-pie %s -o %t.nopie.exe -Wl,-q
RUN: %clang %cflags -fpic -pie %s -o %t.pie.exe -Wl,-q

RUN: llvm-bolt %t.nopie.exe --lite=0 -o %t.nopie --hugify
RUN: llvm-bolt %t.pie.exe --lite=0 -o %t.pie --hugify

RUN: %t.nopie | FileCheck %s -check-prefix=CHECK-NOPIE

CHECK-NOPIE: Hello world

RUN: %t.pie | FileCheck %s -check-prefix=CHECK-PIE

CHECK-PIE: Hello world

*/
