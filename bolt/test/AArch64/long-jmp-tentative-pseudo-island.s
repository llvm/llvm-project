## Check that a function that is not emitted does not affect the alignment
## the LongJmp tentative layout assumes for its code section. The linker veneer
## __AArch64AbsLongThunk_U is eliminated, so it is not emitted, and its constant
## island must not raise the alignment of .text.cold to 16. With
## --align-functions=4, .text.cold is only 4-byte aligned and starts right after
## the 12-byte H in .text, so U is not 16-byte aligned.

# REQUIRES: asserts

# RUN: llvm-mc -filetype=obj -triple=aarch64-linux %s -o %t.o
# RUN: ld.lld %t.o -o %t.exe -q --entry=H
# RUN: echo H > %t.order
# RUN: llvm-bolt %t.exe -o %t.bolt --function-order=%t.order \
# RUN:   --align-functions=4 -debug-only=longjmp > %t.log 2>&1
# RUN: echo "EMITTED:" >> %t.log
# RUN: llvm-nm %t.bolt >> %t.log
# RUN: FileCheck %s --input-file=%t.log
# RUN: FileCheck %s --input-file=%t.log --check-prefix=VENEER

# VENEER: BOLT-INFO: number of removed linker-inserted veneers: 1

# CHECK-DAG: {{^}}H tentative: [[#%x,H:]]
# CHECK-DAG: {{^}}U tentative: [[#%x,U:]]
# CHECK:     EMITTED:
# CHECK-DAG: [[#%.16x,H]] T H{{$}}
# CHECK-DAG: [[#%.16x,U]] T U{{$}}

  .text
  .globl H
  .type H, %function
H:
  add x0, x0, #1
  add x0, x0, #1
  ret
  .size H, .-H

  .type __AArch64AbsLongThunk_U, %function
__AArch64AbsLongThunk_U:
  ldr x16, 1f
  br x16
1:
  .xword U
  .size __AArch64AbsLongThunk_U, .-__AArch64AbsLongThunk_U

  .globl U
  .type U, %function
U:
  ret
  .size U, .-U

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
