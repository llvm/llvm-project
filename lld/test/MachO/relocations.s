# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump -d %t | FileCheck %s

# CHECK: leaq 17(%rip), %rsi

.section __TEXT,__text
.globl _main
_main:
  movl $0x2000004, %eax # write() syscall
  mov $1, %rdi # stdout
  leaq str(%rip), %rsi
  mov $13, %rdx # length of str
  syscall
  mov $0, %rax
  ret

.section __TEXT,__cstring
str:
  .asciz "Hello world!\n"
