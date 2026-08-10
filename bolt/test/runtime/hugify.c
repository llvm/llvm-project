// Make sure BOLT correctly processes --hugify option

#include <stdio.h>

int main(int argc, char **argv) {
  printf("Hello world\n");
  return 0;
}

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags -no-pie %s -o %t.nopie.exe -Wl,-q
RUN: %clang %cflags -fpic %s -o %t.pie.exe -Wl,-q

RUN: llvm-bolt %t.nopie.exe --lite=0 -o %t.nopie --hugify
RUN: llvm-bolt %t.pie.exe --lite=0 -o %t.pie --hugify

RUN: llvm-nm --numeric-sort --print-armap %t.nopie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.nopie | FileCheck %s -check-prefix=CHECK-NOPIE

RUN: llvm-nm --numeric-sort --print-armap %t.pie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.pie | FileCheck %s -check-prefix=CHECK-PIE

CHECK-NM:       W  __hot_start
CHECK-NM-NEXT:  T _start
CHECK-NM:       T main
CHECK-NM:       W __hot_end
CHECK-NM:       t __bolt_hugify_start_program
CHECK-NM-NEXT:  W __bolt_runtime_start

CHECK-NOPIE: Hello world

CHECK-PIE: Hello world

*/
