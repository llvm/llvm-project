# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { \
# RUN:  . = 0x12341; \
# RUN:  .data : { *(.data) } \
# RUN:  . = . + 0x10000; \
# RUN:  .text : { *(.text) } \
# RUN: }" > %t.script

# RUN: ld.lld %t --script %t.script -o %t2
# RUN: llvm-readobj -s %t2 | FileCheck %s

# CHECK:      Sections [
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 0
# CHECK-NEXT:     Name:  (0)
# CHECK-NEXT:     Type: SHT_NULL
# CHECK-NEXT:     Flags [
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x0
# CHECK-NEXT:     Offset: 0x0
# CHECK-NEXT:     Size: 0
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 0
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 1
# CHECK-NEXT:     Name: .data
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       SHF_ALLOC
# CHECK-NEXT:       SHF_WRITE
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x12341
# CHECK-NEXT:     Offset: 0x158
# CHECK-NEXT:     Size: 8
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 1
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }
# CHECK-NEXT:   Section {
# CHECK-NEXT:     Index: 2
# CHECK-NEXT:     Name: .text
# CHECK-NEXT:     Type: SHT_PROGBITS
# CHECK-NEXT:     Flags [
# CHECK-NEXT:       SHF_ALLOC
# CHECK-NEXT:       SHF_EXECINSTR
# CHECK-NEXT:     ]
# CHECK-NEXT:     Address: 0x2234C
# CHECK-NEXT:     Offset: 0x160
# CHECK-NEXT:     Size: 1
# CHECK-NEXT:     Link: 0
# CHECK-NEXT:     Info: 0
# CHECK-NEXT:     AddressAlignment: 4
# CHECK-NEXT:     EntrySize: 0
# CHECK-NEXT:   }

# RUN: echo "SECTIONS { \
# RUN:  . = 0x12Q41; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=NUMERR %s
# NUMERR: malformed number: 0x12Q41

.globl _start;
_start:
nop

.section .data
.quad 0

