# REQUIRES: x86
# Test that jmp instructions (tail calls) also get thunks when the target
# is out of range. Both call and jmp use R_X86_64_PLT32 relocations, so
# thunks must handle jmp identically to call.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10000: { *(.text_low) } \
# RUN:       .text_high 0x200000000: { *(.text_high) } \
# RUN:       }' > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## The jmp to high_target should go through a thunk, just like a call would.
# CHECK:      <_start>:
# CHECK-NEXT:   jmp {{.*}} <__X86_64LongThunk_high_target>

# CHECK:      <__X86_64LongThunk_high_target>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

# CHECK:      <high_target>:
# CHECK-NEXT:   retq

.section .text_low,"ax",@progbits
.globl _start
_start:
  jmp high_target

.section .text_high,"ax",@progbits
.globl high_target
high_target:
  ret
