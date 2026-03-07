# REQUIRES: x86
# Test that calls to undefined weak symbols (which resolve to 0) get thunks
# when the caller is placed far from address 0. An undefined weak symbol
# has a VA of 0, so a call from a high address needs a thunk.

# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text 0x200000000: { *(.text) } \
# RUN:       }' > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## _start is at 8GiB, weak_sym resolves to 0. The displacement overflows
## 32-bit range, so a thunk is needed.
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_weak_sym>
# CHECK-NEXT:   retq

# CHECK:      <__X86_64LongThunk_weak_sym>:
# CHECK-NEXT:   movabsq
# CHECK-NEXT:   leaq
# CHECK-NEXT:   addq    %r10, %r11
# CHECK-NEXT:   jmpq    *%r11

.text
.globl _start
_start:
  call weak_sym
  ret

.weak weak_sym
