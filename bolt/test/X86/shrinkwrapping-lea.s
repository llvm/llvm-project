## This checks that shrink wrapping correctly drops moving push/pops when
## there is an LEA instruction.


# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: ld.lld %t.o -o %t.exe -q
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:     -frame-opt=all -simplify-conditional-tail-calls=false \
# RUN:     -experimental-shrink-wrapping \
# RUN:     -eliminate-unreachable=false | FileCheck %s
# RUN: llvm-objdump -d %t.out --print-imm-hex | \
# RUN:   FileCheck --check-prefix CHECK-OBJDUMP %s

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push  %rbp
  mov   %rsp, %rbp
  push  %rbx
  push  %r14
  subq  $0x20, %rsp
  je  b
c:
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
b:
  je  f
  jmp *JT(,%rdi,8)
d:
  mov %r14, %rdi
  mov %rbx, %rdi
  leaq -0x20(%rbp), %r14
  movq -0x20(%rbp), %rdi
f:
  addq  $0x20, %rsp
  pop %r14
  pop %rbx
  pop %rbp
  ret
  .cfi_endproc
  .size _start, .-_start
  .data
JT:
  .quad c
  .quad d
  .quad f


# CHECK:   BOLT-INFO: Shrink wrapping moved 2 spills inserting load/stores and 0 spills inserting push/pops

## Checks that offsets of instructions accessing the stack were not changed
# CHECK-OBJDUMP:     <_start>:
# CHECK-OBJDUMP:         movq    %rbx, %rdi
# CHECK-OBJDUMP-NEXT:    leaq    -0x20(%rbp), %r14
# CHECK-OBJDUMP:         movq    -0x20(%rbp), %rdi
