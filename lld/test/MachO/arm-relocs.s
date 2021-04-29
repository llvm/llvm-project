# REQUIRES: arm
# RUN: llvm-mc -filetype=obj -triple=armv7-apple-watchos %s -o %t.o
# RUN: ld -dylib -arch armv7 -lSystem -o %t %t.o
# RUN: show-relocs %t.o
# RUN: llvm-objdump --macho -d %t
# RUN: llvm-objdump --triple=thumb -d %t

.globl _arm, _thumb
.syntax unified
.thumb_func _thumb

.p2align 2

.code 16
_thumb:
  bl _arm
  blx _arm
  bl _thumb
  blx _thumb

.code 32
_arm:
  bl _thumb
  blx _thumb
  bl _arm
  blx _arm
