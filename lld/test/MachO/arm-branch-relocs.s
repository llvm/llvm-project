# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7-apple-watchos %s -o %t.o
# RUN: %lld-watchos -dylib -arch armv7 -lSystem -o %t %t.o
# RUN: llvm-objdump --macho -d %t | FileCheck %s

# CHECK:      _arm:
# CHECK-NEXT: blx	_thumb_1
# CHECK-NEXT: blx	_thumb_2
# CHECK-NEXT: bl	_arm
# CHECK-NEXT: bl	_arm

.globl _arm, _thumb_1, _thumb_2
.syntax unified
.thumb_func _thumb_1
.thumb_func _thumb_2

.p2align 2

.code 16
## These two thumb functions are exactly 2 bytes apart in order to test that we
## set the H bit correctly in the BLX instruction.
_thumb_1:
  nop

_thumb_2:
  nop

.code 32
_arm:
  blx _thumb_1
  blx _thumb_2
  bl _arm
  blx _arm
