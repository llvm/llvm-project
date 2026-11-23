# This test checks that BOLT can add BTI to targets at patched entries.

# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -pie %t.o -o %t.exe -nostdlib -Wl,-q,-z,force-bti
# RUN: llvm-bolt %t.exe -o %t.bolt --use-old-text=0 --lite=0 --force-patch | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -dz %t.bolt | FileCheck %s

# CHECK-BOLT: binary is using BTI

# CHECK: <pathedEntries.org.0>:
# CHECK-NEXT: adrp x16, 0x[[#%x,ADRP:]]
# CHECK-NEXT: add x16, x16, #0x[[#%x,ADD:]]
# CHECK-NEXT: br x16

# CHECK: [[#ADRP + ADD]] <pathedEntries>:
# CHECK-NEXT: [[#ADRP + ADD]]: {{.*}} bti c
# CHECK-NEXT: ret

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
  ret
.size _start, .-_start
