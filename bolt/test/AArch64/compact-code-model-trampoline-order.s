## Check that branch relaxation models trampolines added after the same basic
## block in their insertion order. Otherwise, a reused trampoline can appear in
## range of a TBZ branch while being 4 bytes out of range in the output.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld --emit-relocs %t.o -o %t.exe
# RUN: llvm-bolt %t.exe -o %t.bolt --data %t.fdata --split-functions \
# RUN:   --split-all-cold --keep-nops --compact-code-model
# RUN: llvm-objdump -d --disassemble-symbols=_start %t.bolt | FileCheck %s

  .text
  .globl _start
  .type _start, %function
_start:
# FDATA: 0 [unknown] 0 1 _start 0 0 100
  .cfi_startproc
.T0:
  cbnz x3, .TC
# FDATA: 1 _start #.T0# 1 _start #.TC# 0 50
# FDATA: 1 _start #.T0# 1 _start #.T1# 0 50
.T1:
## Both targets are cold. The unconditional branch is split into a trampoline
## first, and the trampoline for the conditional branch is placed after it.
  tbz w0, #0, .LX
  b .LZ
# CHECK:      <_start>:
# CHECK-NEXT: cbnz x3
# CHECK-NEXT: tbz w0, #0x0, 0x[[#%x,TRAMPX:]]
# CHECK-NEXT: b 0x[[#%x,Z:]] <_start.cold.0
# CHECK-NEXT: [[#TRAMPX]]: {{.*}} b {{.*}} <_start.cold.0

.LX:
  ret x30

.TC:
  .rept 8191
  nop
  .endr
## The first trampoline for .LZ is 32772 bytes back, out of TBZ range.
.T2:
  tbz w1, #1, .LZ
# FDATA: 1 _start #.T2# 1 _start #.T3# 0 50
# CHECK:      tbz w1, #0x1, 0x[[#%x,TRAMPZ:]]
# CHECK-NEXT: ret
# CHECK-NEXT: [[#TRAMPZ]]: {{.*}} b 0x[[#Z]] <_start.cold.0
.T3:
  ret x30

.LZ:
  ret x30
  .cfi_endproc
  .size _start, .-_start
