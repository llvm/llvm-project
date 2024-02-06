// This tests checks the minimum alignment of the AARch64 function
// is equal to 4. Otherwise the jitlinker would fail to link the
// binary since the size of the first function after reorder is not
// not a multiple of 4.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0 \
# RUN:   --align-functions-max-bytes=1 \
# RUN:   --data %t.fdata --reorder-functions=exec-count
# RUN: llvm-nm -n %t.bolt | FileCheck %s

# CHECK: {{0|4|8|c}} T dummy
# CHECK-NEXT: {{0|4|8|c}} T _start

  .text
  .align 4
  .global _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 1
   bl dymmy
   ret
  .size _start, .-_start

  .global dummy
  .type dummy, %function
dummy:
# FDATA: 0 [unknown] 0 1 dummy 0 0 42
  adr x0, .Lci
  ret
.Lci:
  .byte 0
  .size dummy, .-dummy
