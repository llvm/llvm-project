# REQUIRES: x86
# Test x86-64 range extension thunks with -pie (PIC mode).
# Verify the same PC-relative thunks are emitted for position-independent
# executables.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10000: { *(.text_low) } \
# RUN:       .text_high 0x200000000: { *(.text_high) } \
# RUN:       }' > %t.script
# RUN: ld.lld -pie -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## Two calls to high_target reuse the same long thunk.
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_high_target>
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_high_target>
# CHECK-NEXT:   retq

## PC-relative long thunk: movabsq + leaq + addq + jmpq sequence.
# CHECK:      <__X86_64LongThunk_high_target>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

## high_target calls back to _start; needs a long thunk.
# CHECK:      <high_target>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk__start>
# CHECK-NEXT:   retq

# CHECK:      <__X86_64LongThunk__start>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

.section .text_low,"ax",@progbits
.globl _start
_start:
  call high_target
  call high_target
  ret

.section .text_high,"ax",@progbits
.globl high_target
high_target:
  call _start
  ret
