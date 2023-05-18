# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: not %lld -arch arm64 %t.o -o %t 2>&1 | FileCheck %s --implicit-check-not=_byte \
# RUN:   --implicit-check-not=_correct

# CHECK-DAG: error: {{.*}}:(symbol _main+0x4): 16-bit LDR/STR to 0x[[#%X,]] (_half) is not 2-byte aligned
# CHECK-DAG: error: {{.*}}:(symbol _main+0xc): 32-bit LDR/STR to 0x[[#%X,]] (_word) is not 4-byte aligned
# CHECK-DAG: error: {{.*}}:(symbol _main+0x14): 64-bit LDR/STR to 0x[[#%X,]] (_double) is not 8-byte aligned
# CHECK-DAG: error: {{.*}}:(symbol _main+0x1c): 128-bit LDR/STR to 0x[[#%X,]] (_quad) is not 16-byte aligned

.globl _main
_main:
  adrp x0, _half@PAGE
  ldrh w0, [x0, _half@PAGEOFF]

  adrp x1, _word@PAGE
  ldr  w1, [x1, _word@PAGEOFF]

  adrp x2, _double@PAGE
  ldr  x2, [x2, _double@PAGEOFF]

  adrp x3, _quad@PAGE
  ldr  q0, [x3, _quad@PAGEOFF]

  adrp x4, _byte@PAGE
  ldrb w4, [x4, _byte@PAGEOFF]

  adrp x5, _correct@PAGE
  ldr  x5, [x5, _correct@PAGEOFF]

.data
.p2align 4
_correct:
.8byte 0
.byte 0
_half:
.2byte 0
_word:
.4byte 0
_double:
.8byte 0
_quad:
.8byte 0
.8byte 0
_byte:
.byte 0
