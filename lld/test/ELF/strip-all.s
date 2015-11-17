# REQUIRES: x86

# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: ld.lld2 %t.o -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix BEFORE
#BEFORE:       .symtab
#BEFORE-NEXT:  .shstrtab
#BEFORE-NEXT:  .strtab

#RUN: ld.lld2 %t.o --strip-all -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix AFTER
#AFTER-NOT: .symtab
#AFTER:     .shstrtab
#AFTER-NOT: .strtab

# Test alias -s
#RUN: ld.lld2 %t.o -s -o %t1
#RUN: llvm-objdump -section-headers %t1 | FileCheck %s -check-prefix AFTER

# exits with return code 42 on linux
.globl _start;
_start:
  mov $60, %rax
  mov $42, %rdi
  syscall
