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
RUN: not llvm-bolt %t.nopie.exe --lite=0 -o /dev/null --hugify \
RUN:   --no-huge-pages=true 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR
RUN: llvm-bolt %t.pie.exe --lite=0 -o %t.pie --hugify
RUN: not llvm-bolt %t.pie.exe --lite=0 -o /dev/null -hugify \
RUN:   -no-huge-pages=true 2>&1 | FileCheck %s -check-prefix=CHECK-ERROR

RUN: llvm-nm --numeric-sort --print-armap %t.nopie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.nopie | FileCheck %s -check-prefix=CHECK
RUN: llvm-readelf -lS %t.nopie | FileCheck %s -check-prefix=CHECK-ALIGNMENT

RUN: llvm-nm --numeric-sort --print-armap %t.pie | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t.pie | FileCheck %s -check-prefix=CHECK
RUN: llvm-readelf -lS %t.pie | FileCheck %s -check-prefix=CHECK-ALIGNMENT

CHECK-NM:       W  __hot_start
CHECK-NM-NEXT:  T _start
CHECK-NM:       T main
CHECK-NM:       W __hot_end
CHECK-NM:       t __bolt_hugify_start_program
CHECK-NM-NEXT:  W __bolt_runtime_start

COM: .text section must have a hugepage alignment
COM: at least one R-E segment must have a hugepage alignment
CHECK-ALIGNMENT: {{.*}} .text PROGBITS {{[0-9a-f]+}}00000 {{.*}} 2097152
CHECK-ALIGNMENT: LOAD 0x{{.*}} R E 0x200000

CHECK: Hello world

CHECK-ERROR: BOLT-ERROR: a conflict between --hugify and --no-huge-pages options

*/
