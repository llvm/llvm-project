# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
# RUN: echo "SECTIONS { .foo : { *(.foo) } .bar : { *(.got.plt) BYTE(0x11) }}" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-readobj -s %t | FileCheck %s

## We have ".got.plt" synthetic section with SHF_ALLOC|SHF_WRITE flags.
## It is empty, so linker removes it, but it have to keep ".got.plt" output
## section because of BYTE command. Here we check that result output section
## gets the same flags as previous allocatable section and does not get
## SHF_WRITE flag from removed syntethic input section.

# CHECK:     Section {
# CHECK:       Index: 2
# CHECK:       Name: .bar
# CHECK-NEXT:  Type: SHT_PROGBITS
# CHECK-NEXT:  Flags [
# CHECK-NEXT:    SHF_ALLOC
# CHECK-NEXT:    SHF_EXECINSTR
# CHECK-NEXT:  ]

## Check flags are the same if we omit empty synthetic section in script.
# RUN: echo "SECTIONS { .foo : { *(.foo) } .bar : { BYTE(0x11) }}" > %t.script
# RUN: ld.lld -o %t --script %t.script %t.o
# RUN: llvm-readobj -s %t | FileCheck %s

.section .foo,"ax"
.quad 0
