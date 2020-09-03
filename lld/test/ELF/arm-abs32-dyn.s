// REQUIRES: arm
// RUN: llvm-mc -filetype=obj -triple=armv7a-none-linux %s -o %t.o

/// Creates a R_ARM_ABS32 relocation against foo and bar, bar has hidden
/// visibility so we expect a R_ARM_RELATIVE
 .syntax unified
 .globl foo
foo:
 .globl bar
 .hidden bar
bar:

 .data
 .word foo
 .word bar

/// In PIC mode, if R_ARM_TARGET1 represents R_ARM_ABS32 (the default), an
/// R_ARM_TARGET1 to a non-preemptable symbol also creates an R_ARM_RELATIVE in
/// a writable section.
 .word bar(target1)

// RUN: ld.lld -shared -o %t.so %t.o
// RUN: llvm-readobj --symbols --dyn-relocations %t.so | FileCheck %s
// RUN: llvm-readelf -x .data %t.so | FileCheck --check-prefix=HEX %s

// CHECK:      Dynamic Relocations {
// CHECK-NEXT:   0x30204 R_ARM_RELATIVE
// CHECK-NEXT:   0x30208 R_ARM_RELATIVE
// CHECK-NEXT:   0x30200 R_ARM_ABS32 foo 0x0
// CHECK-NEXT: }

// CHECK:      Symbols [
// CHECK:        Symbol {
// CHECK:          Name: bar
// CHECK-NEXT:     Value: 0x101A8

// CHECK:        Symbol {
// CHECK:          Name: foo
// CHECK-NEXT:     Value: 0x101A8

// HEX: 0x00030200 00000000 a8010100 a8010100
