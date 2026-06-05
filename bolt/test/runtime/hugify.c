// Make sure BOLT correctly processes --hugify option

#include <stdio.h>

// The dummy var is used for testing the hugify feature on pre-5.10 kernels,
// which do not include the patch "fs/binfmt_elf: use PT_LOAD p_align values for
// suitable start address".
// The code size of the hugify test with the dummy var is about 2 MB. This
// allows testing a corner case where the left (rounded down) and right (rounded
// up) boundaries are located in different hugepages when the segment is mapped
// with page alignment instead of the expected segment alignment. It is expected
// that the hugified binary (PIE only) doesn't crash with a segfault, even in
// the case of unexpected segment alignment, on pre-5.10 kernels.
const char dummy_const_var_in_text[2 * 1024 * 1024 - 0x1000]
    __attribute__((section(".text")));

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
