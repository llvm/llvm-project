# This checks that frame optimizer does not try to optimize away caller-saved
# regs when we do not have complete aliasing info (when there is an LEA
# instruction and the function does arithmetic with stack addresses).


# REQUIRES: system-linux

# RUN: llvm-mc -filetype=obj -triple x86_64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-strip --strip-unneeded %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q -nostdlib
# RUN: llvm-bolt %t.exe -relocs -o %t.out -data %t.fdata \
# RUN:     -frame-opt=all -simplify-conditional-tail-calls=false \
# RUN:     -lite=0 -eliminate-unreachable=false | FileCheck %s
# RUN: llvm-objdump -d %t.out --print-imm-hex | \
# RUN:   FileCheck --check-prefix CHECK-OBJDUMP %s

  .globl foo
  .type foo, %function
foo:
  .cfi_startproc
  movq $0, (%rsi)
  ret
  .cfi_endproc
  .size foo, .-foo


  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
# FDATA: 0 [unknown] 0 1 _start 0 0 1
  push  %rbp
  mov   %rsp, %rbp
  subq  $0x20, %rsp
  je  b
c:
  addq  $0x20, %rsp
  pop %rbp
  ret
b:
  movq %rdi, %r13
  movq %r13, -0x08(%rbp)
  leaq -0x08(%rbp), %rsi
  callq foo
  movq -0x08(%rbp), %r13
  jmp c
  .cfi_endproc
  .size _start, .-_start


# CHECK:   BOLT-INFO: FOP deleted 0 load(s) (dyn count: 0) and 0 store(s)

# CHECK-OBJDUMP:     <_start>:
# CHECK-OBJDUMP:         movq    %rdi, %r13
# CHECK-OBJDUMP-NEXT:    movq    %r13, -0x8(%rbp)
# CHECK-OBJDUMP-NEXT:    leaq
# CHECK-OBJDUMP-NEXT:    callq
# CHECK-OBJDUMP-NEXT:    movq    -0x8(%rbp), %r13
