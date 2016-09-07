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
# RUN:  . = 0x1 ? 0x16000 : 0x999999; \
# RUN:  .ternary1 : { *(.ternary1) } \
# RUN:  . = 0x0 ? 0x999999 : 0x17000; \
# RUN:  .ternary2 : { *(.ternary2) } \
# RUN:  . = 0x0 < 0x1 ? 0x18000 : 0x999999; \
# RUN:  .less : { *(.less) } \
# RUN:  . = 0x1 <= 0x1 ? 0x19000 : 0x999999; \
# RUN:  .lesseq : { *(.lesseq) } \
# RUN:  . = 0x1 > 0x0 ? 0x20000 : 0x999999; \
# RUN:  .great : { *(.great) } \
# RUN:  . = 0x1 >= 0x1 ? 0x21000 : 0x999999; \
# RUN:  .greateq : { *(.greateq) } \
# RUN:  . = 0x1 == 0x1 ? 0x22000 : 0x999999; \
# RUN:  .eq : { *(.eq) } \
# RUN:  . = 0x2 != 0x1 ? 0x23000 : 0x999999; \
# RUN:  .neq : { *(.neq) } \
# RUN:  . = CONSTANT (MAXPAGESIZE) * 0x24; \
# RUN:  .maxpagesize : { *(.maxpagesize) } \
# RUN:  . = CONSTANT (COMMONPAGESIZE) * 0x25; \
# RUN:  .commonpagesize : { *(.commonpagesize) } \
# RUN:  . = DATA_SEGMENT_ALIGN (CONSTANT (MAXPAGESIZE), CONSTANT (COMMONPAGESIZE)); \
# RUN:  .datasegmentalign : { *(.datasegmentalign) } \
# RUN:  . = DATA_SEGMENT_END (.); \
# RUN:  . = 0x27000; \
# RUN:  . += 0x1000; \
# RUN:  .plusassign : { *(.plusassign) } \
# RUN:  . = ((. + 0x1fff) & ~(0x1000 + -1)); \
# RUN:  .unary : { *(.unary) } \
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
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .ternary1
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x16000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .ternary2
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x17000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .less
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x18000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .lesseq
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x19000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .great
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x20000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .greateq
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x21000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .eq
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x22000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .neq
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x23000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .maxpagesize
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x4800000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize: 0
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .commonpagesize
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x25000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .datasegmentalign
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x200000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .plusassign
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x28000
# CHECK-NEXT:   Offset:
# CHECK-NEXT:   Size:
# CHECK-NEXT:   Link:
# CHECK-NEXT:   Info:
# CHECK-NEXT:   AddressAlignment:
# CHECK-NEXT:   EntrySize:
# CHECK-NEXT: }
# CHECK-NEXT: Section {
# CHECK-NEXT:   Index:
# CHECK-NEXT:   Name: .unary
# CHECK-NEXT:   Type: SHT_PROGBITS
# CHECK-NEXT:   Flags [
# CHECK-NEXT:     SHF_ALLOC
# CHECK-NEXT:   ]
# CHECK-NEXT:   Address: 0x2A000
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
# BRACKETERR: ) expected, but got ;

## Missing opening bracket.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x10000 + 0x1000 + 0x1000) * 0x2; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=BRACKETERR2 %s
# BRACKETERR2: ; expected, but got )

## Empty expression.
# RUN: echo "SECTIONS { \
# RUN:  . = ; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=ERREXPR %s
# ERREXPR: malformed number: ;

## Div by zero error.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x10000 / 0x0; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=DIVZERO %s
# DIVZERO: division by zero

## Broken ternary operator expression.
# RUN: echo "SECTIONS { \
# RUN:  . = 0x1 ? 0x2; \
# RUN: }" > %t.script
# RUN: not ld.lld %t --script %t.script -o %t2 2>&1 | \
# RUN:  FileCheck --check-prefix=TERNERR %s
# TERNERR: : expected, but got ;

.globl _start
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

.section .ternary1, "a"
.quad 0

.section .ternary2, "a"
.quad 0

.section .less, "a"
.quad 0

.section .lesseq, "a"
.quad 0

.section .great, "a"
.quad 0

.section .greateq, "a"
.quad 0

.section .eq, "a"
.quad 0

.section .neq, "a"
.quad 0

.section .maxpagesize, "a"
.quad 0

.section .commonpagesize, "a"
.quad 0

.section .datasegmentalign, "a"
.quad 0

.section .plusassign, "a"
.quad 0

.section .unary, "a"
.quad 0
