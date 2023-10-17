// REQUIRES: arm
/// Create a secure app and import library using CMSE.
/// Create a non-secure app that refers symbols in the import library.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main cmse-implib.s -o implib.o -I%S/Inputs/
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main cmse-secure-app.s -o secureapp.o
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main cmse-non-secure-app.s -o nonsecureapp.o
/// Create the secure app and import library.
// RUN: ld.lld -e secure_entry --section-start .gnu.sgstubs=0x20000 --cmse-implib implib.o secureapp.o --out-implib=implib.lib -o secureapp
/// Link the non-secure app against the import library.
// RUN: ld.lld -e nonsecure_entry -Ttext=0x8000 implib.lib nonsecureapp.o -o nonsecureapp
// RUN: llvm-readelf -s implib.lib | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn secureapp | FileCheck %s --check-prefixes=SECUREDISS
// RUN: llvm-objdump -d --no-show-raw-insn nonsecureapp | FileCheck %s --check-prefixes=NONSECUREDISS

// SECUREDISS-LABEL: <entry>:
// SECUREDISS-NEXT:    20000: sg
// SECUREDISS-NEXT:           b.w {{.*}} <__acle_se_entry>

// SECUREDISS-LABEL: <__acle_se_entry>:
// SECUREDISS-NEXT:    20008: nop

// SECUREDISS-LABEL: <secure_entry>:
// SECUREDISS-NEXT:    2000c: bl {{.*}} <__acle_se_entry>
// SECUREDISS-NEXT:           bx lr

// NONSECUREDISS-LABEL: <nonsecure_entry>:
// NONSECUREDISS-NEXT:  8000: push {r0, lr}
// NONSECUREDISS-NEXT:  bl 0x20000
// NONSECUREDISS-NEXT:  pop.w {r0, lr}
// NONSECUREDISS-NEXT:  bx lr

// CHECK:      Symbol table '.symtab' contains 2 entries:
// CHECK-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// CHECK-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK-NEXT:      1: 00020001     8 FUNC    GLOBAL DEFAULT   ABS entry

//--- cmse-implib.s
  .include "arm-cmse-macros.s"

  .syntax unified
  .text

  cmse_veneer entry, function, global, function, global

//--- cmse-secure-app.s
  .align  2
    // Main entry point.
    .global secure_entry
    .thumb_func
secure_entry:
    bl entry
    bx lr
  .size  secure_entry, .-secure_entry

//--- cmse-non-secure-app.s
  .align  2
  .global nonsecure_entry
  .thumb
  .thumb_func
  .type  nonsecure_entry, %function
nonsecure_entry:
  push {r0,lr}
  bl entry
  pop {r0,lr}
  bx lr
  .size  nonsecure_entry, .-nonsecure_entry
