// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple x86_64-pc-linux-gnu %s -o %t.o
// RUN: echo "SECTIONS { . = SIZEOF_HEADERS; .text : { *(.text) } }" > %t.script
// RUN: ld.lld -T %t.script %t.o -o %t
// RUN: llvm-readobj --symbols %t | FileCheck %s

// Test that _start is in the correct section.
// CHECK:      Name: _start
// CHECK-NEXT: Value:
// CHECK-NEXT: Size: 0
// CHECK-NEXT: Binding: Global
// CHECK-NEXT: Type: None
// CHECK-NEXT: Other: 0
// CHECK-NEXT: Section: dn

// Show that --gc-sections works when there are many sections, and that
// referenced common and absolute symbols in such cases are not removed, nor are
// they incorrectly attributed to the sections with index 0xFFF1 or 0xFFF2.
// RUN: ld.lld %t.o -T %t.script -o %t --gc-sections
// RUN: llvm-readobj --symbols --sections %t | FileCheck %s --check-prefix=GC

// GC:      Sections [
// GC-NEXT:   Section {
// GC-NEXT:     Index: 0
// GC-NEXT:     Name:  (0)
// GC:        Section {
// GC-NEXT:     Index: 1
// GC-NEXT:     Name: dn
// GC:        Section {
// GC-NEXT:     Index: 2
// GC-NEXT:     Name: .bss
// GC:        Section {
// GC-NEXT:     Index: 3
// GC-NEXT:     Name: .comment
// GC:        Section {
// GC-NEXT:     Index: 4
// GC-NEXT:     Name: .symtab
// GC:        Section {
// GC-NEXT:     Index: 5
// GC-NEXT:     Name: .shstrtab
// GC:        Section {
// GC-NEXT:     Index: 6
// GC-NEXT:     Name: .strtab
// GC-NOT:    Section {

// GC:      Symbols [
// GC-NEXT:   Symbol {
// GC-NEXT:     Name:  (0)
// GC:        Symbol {
// GC-NEXT:     Name: sdn
// GC:        Symbol {
// GC-NEXT:     Name: _start
// GC:        Symbol {
// GC-NEXT:     Name: abs
// GC:        Symbol {
// GC-NEXT:     Name: common
// GC-NOT:    Symbol {

.macro gen_sections4 x
        .section a\x,"a"
        .global sa\x
        sa\x:
        .section b\x,"a"
        .global sa\x
        sb\x:
        .section c\x,"a"
        .global sa\x
        sc\x:
        .section d\x,"a"
        .global sa\x
        sd\x:
.endm

.macro gen_sections8 x
        gen_sections4 a\x
        gen_sections4 b\x
.endm

.macro gen_sections16 x
        gen_sections8 a\x
        gen_sections8 b\x
.endm

.macro gen_sections32 x
        gen_sections16 a\x
        gen_sections16 b\x
.endm

.macro gen_sections64 x
        gen_sections32 a\x
        gen_sections32 b\x
.endm

.macro gen_sections128 x
        gen_sections64 a\x
        gen_sections64 b\x
.endm

.macro gen_sections256 x
        gen_sections128 a\x
        gen_sections128 b\x
.endm

.macro gen_sections512 x
        gen_sections256 a\x
        gen_sections256 b\x
.endm

.macro gen_sections1024 x
        gen_sections512 a\x
        gen_sections512 b\x
.endm

.macro gen_sections2048 x
        gen_sections1024 a\x
        gen_sections1024 b\x
.endm

.macro gen_sections4096 x
        gen_sections2048 a\x
        gen_sections2048 b\x
.endm

.macro gen_sections8192 x
        gen_sections4096 a\x
        gen_sections4096 b\x
.endm

.macro gen_sections16384 x
        gen_sections8192 a\x
        gen_sections8192 b\x
.endm

.macro gen_sections32768 x
        gen_sections16384 a\x
        gen_sections16384 b\x
.endm

gen_sections32768 a
gen_sections16384 b
gen_sections8192 c
gen_sections4096 d
gen_sections2048 e
gen_sections1024 f
gen_sections512 g
gen_sections256 h
gen_sections128 i
gen_sections64 j
gen_sections32 k
gen_sections16 l
gen_sections8 m
gen_sections4 n

.global abs
abs = 0x12345678

.comm common,4,4

.global _start
_start:
  .quad abs
  .quad common
