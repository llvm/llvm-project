# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -o /dev/null %t.o 2>&1 | FileCheck %s

# CHECK: error: TLV reloc requires MOVQ instruction

.text
.globl _main
_main:
  leaq _foo@TLVP(%rip), %rax
  ret

.section	__DATA,__thread_vars,thread_local_variables
_foo:
