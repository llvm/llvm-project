# REQUIRES: x86
# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/main.s -o %t/main.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/foo.s -o %t/foo.o
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %t/bar.s -o %t/bar.o

## A positive integer is allowed.
# RUN: %lld --threads=1 %t/main.o %t/foo.o %t/bar.o -o %t/out.1 -final_output out
# RUN: %lld --threads=2 %t/main.o %t/foo.o %t/bar.o -o %t/out.2 -final_output out

## The number of threads must not affect the output: parsing the input files
## and writing the output sections happen in parallel, so check that the
## linked images are byte-identical regardless of the thread count.
# RUN: cmp %t/out.1 %t/out.2

# RUN: not %lld --threads=all %t/main.o -o /dev/null 2>&1 | FileCheck %s -DN=all
# RUN: not %lld --threads=0 %t/main.o -o /dev/null 2>&1 | FileCheck %s -DN=0
# RUN: not %lld --threads=-1 %t/main.o -o /dev/null 2>&1 | FileCheck %s -DN=-1

# CHECK: error: --threads=: expected a positive integer, but got '[[N]]'

#--- main.s
.globl _main
.text
_main:
  callq _foo
  callq _bar
  movq _gvar@GOTPCREL(%rip), %rax
  leaq L_str(%rip), %rcx
  movq (%rax), %rax
  retq

.section __DATA,__data
.globl _ptr
_ptr:
  .quad _foo

.section __TEXT,__cstring
L_str:
  .asciz "main"

#--- foo.s
.globl _foo
.text
_foo:
  callq _bar
  leaq _gvar(%rip), %rax
  retq

.section __DATA,__data
.globl _gvar
_gvar:
  .quad 0x1234

#--- bar.s
.globl _bar
.text
_bar:
  movq _ptr(%rip), %rax
  retq
