# REQUIRES: aarch64
## A thunk section's forward thunks are sorted by descending destination so promoting
## one to its long form cannot push another out of range. In creation order the promotions cascade,
## one per pass, and exceed convergence limit (issue #61250).

# RUN: rm -rf %t && split-file %s %t && cd %t
# RUN: llvm-mc -filetype=obj -triple=aarch64 a.s -o a.o
# RUN: ld.lld -T lds a.o -o out
# RUN: llvm-objdump -d --no-show-raw-insn out | FileCheck %s

## One thunk section holds both directions: backward thunks first, in creation
## order lo0, lo1, lo2 (they need no sorting), then forward thunks by descending
## destination. The farthest forward thunks become long (ldr), nearer ones short (b).
# CHECK:      <__AArch64AbsLongThunk_lo0>:
# CHECK:      <__AArch64AbsLongThunk_lo1>:
# CHECK:      <__AArch64AbsLongThunk_lo2>:
# CHECK:      <__AArch64AbsLongThunk_hi43>:
# CHECK-NEXT:  ldr x16,
# CHECK:      <__AArch64AbsLongThunk_hi35>:
# CHECK-NEXT:  ldr x16,
# CHECK:      <__AArch64AbsLongThunk_hi34>:
# CHECK-NEXT:  b {{.*}} <hi34>
# CHECK:      <__AArch64AbsLongThunk_hi0>:

## getDestVA keys a PLT-routed thunk on the PLT entry, not the symbol value. bar's thunk sorts after
## the backward thunk to local cbwd; the symbol value 0 would place it backward and flip the order.
# RUN: llvm-mc -filetype=obj -triple=aarch64 %p/Inputs/shared.s -o b.o
# RUN: ld.lld -shared b.o -o b.so
# RUN: llvm-mc -filetype=obj -triple=aarch64 c.s -o c.o
# RUN: ld.lld -T lds2 c.o b.so -o out2
# RUN: llvm-objdump -d --no-show-raw-insn out2 | FileCheck %s --check-prefix=PLT
# PLT:      <__AArch64AbsLongThunk_cbwd>:
# PLT:      <__AArch64AbsLongThunk_bar>:

#--- a.s
n = 44

.section .low,"ax",%progbits
.rept 3
.globl lo\+
lo\+:
  ret
  .space 12
.endr

.text
.globl _start
_start:
.rept 3
  bl lo\+
.endr
.rept n
  bl hi\+
  .space 12
.endr

.section .high,"ax",%progbits
.rept n
  .globl hi\+
hi\+:
  ret
  .space 12
.endr

#--- lds
## .low sits far below .text, so lo<n> calls need (always long) backward thunks.
## .high follows a gap tuned (44 matches n in a.s) so that, once the thunk
## section shifts .high, only the farthest hi<i> is out of range; in creation
## order each promotion then pushes out exactly one more.
SECTIONS {
  .low 0x10000 : { *(.low) }
  .text 0x10000000 : { *(.text) }
  . = . + 0x8000000 - (44 - 1) * 16;
  .high : { *(.high) }
}

#--- c.s
.section .cbwd,"ax",%progbits
.globl cbwd
cbwd:
  ret

.text
.globl _start
_start:
  bl bar
  bl cbwd
  ret

#--- lds2
## cbwd is far below and the PLT far above .text, so both calls need thunks.
SECTIONS {
  .cbwd 0x1000 : { *(.cbwd) }
  .text 0x10000000 : { *(.text) }
  .plt 0x20000000 : { *(.plt) }
}
