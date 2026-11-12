# This test checks patch entries functionality

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -pie %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0 --skip-funcs=_start
# RUN: llvm-objdump -dz %t.bolt | FileCheck %s

# CHECK: <pathedEntries.org.0>:
# CHECK-NEXT: adrp x16, 0x[[#%x,ADRP:]]
# CHECK-NEXT: add x16, x16, #0x[[#%x,ADD:]]
# CHECK-NEXT: br x16

# CHECK: [[#ADRP + ADD]] <pathedEntries>:
# CHECK-NEXT: [[#ADRP + ADD]]: {{.*}} ret

.text
.balign 4
.global pathedEntries
.type pathedEntries, %function
pathedEntries:
  .rept 32
  nop
  .endr
  ret
.size pathedEntries, .-pathedEntries

.global _start
.type _start, %function
_start:
  bl pathedEntries
  .inst 0xdeadbeef
  ret
.size _start, .-_start
