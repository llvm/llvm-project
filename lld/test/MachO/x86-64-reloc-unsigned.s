# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: %lld -o %t %t.o
# RUN: llvm-objdump --macho --rebase --full-contents %t | FileCheck %s

# RUN: %lld -pie -o %t-pie %t.o
# RUN: llvm-objdump --macho --rebase %t-pie | FileCheck %s --check-prefix=PIE
# RUN: %lld -pie -no_pie -o %t-no-pie %t.o
# RUN: llvm-objdump --macho --rebase %t-no-pie | FileCheck %s --check-prefix=NO-PIE
# RUN: %lld -no_pie -pie -o %t-pie %t.o
# RUN: llvm-objdump --macho --rebase %t-pie | FileCheck %s --check-prefix=PIE

# CHECK:       Contents of section __DATA,foo:
# CHECK-NEXT:  100001000 08100000 01000000
# CHECK:       Contents of section __DATA,bar:
# CHECK-NEXT:  100001008 011000f0 11211111 02000000

# PIE:      Rebase table:
# PIE-NEXT: segment  section            address           type
# PIE-DAG:  __DATA   foo                0x[[#%X,ADDR:]]   pointer
# PIE-DAG:  __DATA   bar                0x[[#ADDR + 8]]   pointer
# PIE-DAG:  __DATA   bar                0x[[#ADDR + 12]]  pointer
# PIE-DAG:  __DATA   baz                0x[[#ADDR + 20]]  pointer

# NO-PIE:      Rebase table:
# NO-PIE-NEXT: segment  section            address           type
# NO-PIE-EMPTY:

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

.section __DATA,baz
## Generates a section relocation.
.quad L_.baz
L_.baz:
  .space 0

.text
_main:
  mov $0, %rax
  ret
