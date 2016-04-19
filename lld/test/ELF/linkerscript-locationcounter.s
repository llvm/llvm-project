# REQUIRES: x86
# RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t
# RUN: echo "SECTIONS { \
# RUN:  . = 0xFFF0; \
# RUN:  . = . + 0x10; \
# RUN:  .plus : { *(.plus) } \
# RUN:  . = 0x11010 - 0x10; \
# RUN:  .minus : { *(.minus) } \
# RUN:  . = 0x24000 / 0x2; \
# RUN:  .div : { *(.div) } \
# RUN:  . = 0x11000 + 0x1000 * 0x2; \
# RUN:  .mul : { *(.mul) } \
# RUN:  . = 0x10000 + (0x1000 + 0x1000) * 0x2; \
# RUN:  .bracket : { *(.bracket) } \
# RUN:  . = 0x17000 & 0x15000; \
# RUN:  .and : { *(.and) } \
# RUN: }" > %t.script
# RUN: ld.lld %t --script %t.script -o %t2
# RUN: llvm-readobj -s %t2 | FileCheck %s

# CHECK: Section {
# CHECK:   Index: 1
# CHECK:   Name: .plus
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x10000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index: 2
# CHECK-NEXT:   Name: .minus
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x11000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index: 3
# CHECK-NEXT:   Name: .div
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x12000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index: 4
# CHECK-NEXT:   Name: .mul
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x13000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index: 5
# CHECK-NEXT:   Name: .bracket
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x14000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .and
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x15000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }

## Mailformed number error.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x12Q41; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=NUMERR %s
# NUMERR: malformed number: 0x12Q41

## Missing closing bracket.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x10000 + (0x1000 + 0x1000 * 0x2; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=BRACKETERR %s
# BRACKETERR: ) expected

## Missing opening bracket.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x10000 + 0x1000 + 0x1000) * 0x2; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=BRACKETERR2 %s
# BRACKETERR2: stray token: )

## Empty expression.
# RUN: echo "SECTIONS { \
# RUN:  . = ; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=ERREXPR %s
# ERREXPR: error in location counter expression

## Div by zero error.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x10000 / 0x0; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=DIVZERO %s
# DIVZERO: division by zero

.globl _start;
_start:
nop

.section .plus, "a"
.quad 0

.section .minus, "a"
.quad 0

.section .div, "a"
.quad 0

.section .mul, "a"
.quad 0

.section .bracket, "a"
.quad 0

.section .and, "a"
.quad 0
