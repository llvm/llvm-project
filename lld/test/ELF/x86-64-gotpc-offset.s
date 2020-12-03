# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t.o
# RUN: ld.lld %t.o -o %t
# RUN: llvm-objdump -s -d %t | FileCheck %s

# CHECK:      Contents of section .got:
# CHECK-NEXT: {{^}} [[#%x,ADDR:]] {{.*}} 00000000

# CHECK:      leal {{.*}}(%rip), %eax  # {{.*}} <foo>
# CHECK-NEXT: movl {{.*}}(%rip), %eax  # [[#ADDR+4]]
# CHECK-NEXT: movq {{.*}}(%rip), %rax  # [[#ADDR+1]]

## movl foo@GOTPCREL(%rip), %eax
  movl 0(%rip), %eax
  .reloc .-4, R_X86_64_GOTPCRELX, foo-4

## The instruction has an offset (addend!=-4). It is incorrect to relax movl to leal.
## movl foo@GOTPCREL+4(%rip), %eax
  movl 0(%rip), %eax
  .reloc .-4, R_X86_64_GOTPCRELX, foo

## This does not make sense because it loads one byte past the GOT entry.
## It is just to demonstrate the behavior.
## movq foo@GOTPCREL+1(%rip), %rax
  movq 0(%rip), %rax
  .reloc .-4, R_X86_64_REX_GOTPCRELX, foo-3

.globl foo
foo:
  nop
