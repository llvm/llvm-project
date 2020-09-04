# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: lld -flavor darwinnew -o %t %t.o
# RUN: llvm-objdump --full-contents %t | FileCheck %s
# CHECK: Contents of section __DATA,foo:
# CHECK:  100001000 08100000 01000000
# CHECK: Contents of section __DATA,bar:
# CHECK:  100001008 011000f0 11211111 02000000

.globl _main, _foo, _bar

.section __DATA,foo
_foo:
.quad _bar

.section __DATA,bar
_bar:
## We create a .int symbol reference here -- with non-zero data immediately
## after -- to check that lld reads precisely 32 bits (and not more) of the
## implicit addend when handling unsigned relocations of r_length = 2.
## Note that __PAGEZERO occupies the lower 32 bits, so all symbols are above
## that. To get a final relocated address that fits within 32 bits, we need to
## subtract an offset here.
.int _foo - 0x0fffffff
## The unsigned relocation should support 64-bit addends too (r_length = 3).
.quad _foo + 0x111111111

.text
_main:
  mov $0, %rax
  ret
