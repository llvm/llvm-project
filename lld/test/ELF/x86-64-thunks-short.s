# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: echo 'SECTIONS { \
# RUN:       .text_low 0x10000: { *(.text_low) } \
# RUN:       .text_high 0x80010000: { *(.text_high) } \
# RUN:       }' > %t.script
# RUN: ld.lld -T %t.script %t.o -o %t
# RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s

## The thunk is placed just after _start's code. The call from _start
## (at 0x10000) to high_target (at 0x8001000e) barely overflows signed
## 32-bit range, but the thunk (at 0x1000b, a few bytes closer) can
## still reach high_target with a 5-byte jmp rel32.

## Both calls to high_target reuse the same short thunk.
# CHECK:      <_start>:
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_high_target>
# CHECK-NEXT:   callq {{.*}} <__X86_64LongThunk_high_target>
# CHECK-NEXT:   retq

## Short thunk: a plain jmp rel32 (5 bytes), NOT the movabsq sequence.
# CHECK:      <__X86_64LongThunk_high_target>:
# CHECK-NEXT:   jmp{{.*}} <high_target>
# CHECK-NOT:    movabsq

# CHECK:      <high_target>:
# CHECK-NEXT:   retq

.section .text_low,"ax",@progbits
.globl _start
_start:
  call high_target
  call high_target
  ret

.section .text_high,"ax",@progbits
## 14 bytes of padding so high_target is just past the 2GiB range of the
## call in _start, but still within 2GiB of the thunk placed after _start.
.space 14
.globl high_target
high_target:
  ret
