# REQUIRES: x86
# Test that -z retpolineplt works together with range extension thunks.
# Retpoline changes the PLT format but the thunk mechanism (which bypasses
# PLT for non-preemptible symbols) should still function correctly.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10000: { *(.text_low) } \
# RUN:       .text_high 0x200000000: { *(.text_high) } \
# RUN:       }' > %t.script
# RUN: ld.lld -z retpolineplt -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## The call to high_target needs a thunk. Even with -z retpolineplt,
## non-preemptible direct calls should get a regular thunk (not go
## through the retpoline PLT).
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_high_target>
# CHECK-NEXT:   retq

# CHECK:      <__X86_64LongThunk_high_target>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

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
  ret

.section .text_high,"ax",@progbits
.globl high_target
high_target:
  call _start
  ret
