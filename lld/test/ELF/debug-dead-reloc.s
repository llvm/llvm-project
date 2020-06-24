# REQUIRES: x86
## Test we resolve symbolic relocations in .debug_* sections to a tombstone
## value if the referenced symbol is discarded (--gc-sections, non-prevailing
## section group, SHF_EXCLUDE, /DISCARD/, etc).

# RUN: echo '.globl _start; _start: call group' | llvm-mc -filetype=obj -triple=x86_64 - -o %t.o
# RUN: llvm-mc -filetype=obj -triple=x86_64 %s -o %t1.o
# RUN: ld.lld --gc-sections %t.o %t1.o %t1.o -o %t
# RUN: llvm-objdump -s %t | FileCheck %s

# CHECK:      Contents of section .debug_loc:
# CHECK-NEXT:  0000 feffffff ffffffff feffffff ffffffff
# CHECK-NEXT: Contents of section .debug_ranges:
# CHECK-NEXT:  0000 feffffff ffffffff feffffff ffffffff
# CHECK-NEXT: Contents of section .debug_addr:
# CHECK-NEXT:  0000 {{.*}}000 00000000 {{.*}}000 00000000
# CHECK-NEXT:  0010 ffffffff  ffffffff {{.*}}000 00000000
# CHECK-NEXT: Contents of section .debug_foo:
# CHECK-NEXT:  0000 ffffffff ffffffff 08000000 00000000
# CHECK-NEXT:  0010 ffffffff ffffffff 08000000 00000000

.section .text.1,"ax"
  .byte 0
.section .text.2,"axe"
  .byte 0
.section .text.3,"axG",@progbits,group,comdat
.globl group
group:
  .byte 0

## Resolved to UINT64_C(-2), with the addend ignored.
## UINT64_C(-1) is a reserved value (base address selection entry) which can't be used.
.section .debug_loc
  .quad .text.1+8
.section .debug_ranges
  .quad .text.2+16

.section .debug_addr
## .text.3 is a local symbol. The symbol defined in a non-prevailing group is
## discarded. Resolved to UINT64_C(-1).
  .quad .text.3+24
## group is a non-local symbol. The relocation from the second %t1.o gets
## resolved to the prevailing copy.
  .quad group+32

.section .debug_foo
  .quad .text.1+8

## We only deal with DW_FORM_addr. Don't special case short-range absolute
## relocations. Treat them like regular absolute relocations referencing
## discarded symbols, which are resolved to the addend.
  .long .text.1+8
  .long 0
