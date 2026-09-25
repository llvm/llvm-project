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
RUN: llvm-bolt %t.nopie.exe --lite=0 -o %t.nopie.all-text --hugify \
RUN:   --hugify-all-text
RUN: not llvm-bolt %t.nopie.exe --lite=0 -o %t.nopie.all-text.err \
RUN:   --hugify-all-text 2>&1 | \
RUN:   FileCheck %s -check-prefix=CHECK-HUGIFY-ALL-TEXT-ERR

RUN: llvm-nm --numeric-sort --print-armap %t.nopie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.nopie | FileCheck %s -check-prefix=CHECK-NOPIE

RUN: llvm-nm --numeric-sort --print-armap %t.pie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.pie | FileCheck %s -check-prefix=CHECK-PIE
RUN: %t.nopie.all-text | FileCheck %s -check-prefix=CHECK-NOPIE
RUN: llvm-readelf -x .bolt.hugify.config %t.nopie | \
RUN:   FileCheck %s -check-prefix=CHECK-HUGIFY-HOT
RUN: llvm-readelf -x .bolt.hugify.config %t.nopie.all-text | \
RUN:   FileCheck %s -check-prefix=CHECK-HUGIFY-ALL-TEXT

CHECK-NM:       W  __hot_start
CHECK-NM-NEXT:  T _start
CHECK-NM:       T main
CHECK-NM:       W __hot_end
CHECK-NM:       t __bolt_hugify_start_program
CHECK-NM-NEXT:  W __bolt_runtime_start

CHECK-NOPIE: Hello world

CHECK-PIE: Hello world

CHECK-HUGIFY-HOT: 0x{{[0-9a-f]+}} 00000000

CHECK-HUGIFY-ALL-TEXT: 0x{{[0-9a-f]+}} 01000000

CHECK-HUGIFY-ALL-TEXT-ERR: BOLT-ERROR: --hugify-all-text requires --hugify

*/
