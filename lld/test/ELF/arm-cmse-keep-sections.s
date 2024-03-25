// REQUIRES: arm
/// Create a secure app and import library using CMSE.
/// Create a non-secure app that refers symbols in the import library.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main cmse-implib.s -o implib.o -I%S/Inputs/
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main cmse-secure-app.s -o secureapp.o
/// Create the secure app and import library.
// RUN: ld.lld -e secure_entry --section-start .gnu.sgstubs=0x1000000 --section-start SECURE1=0x10 --section-start SECURE2=0x2000000 --cmse-implib implib.o secureapp.o --out-implib=implib.lib -o secureapp --gc-sections
// RUN: llvm-readelf -s implib.lib | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn secureapp | FileCheck %s --check-prefix=DISS


// DISS-LABEL: <__acle_se_entry1>:
// DISS-NEXT:  10: nop

// DISS-LABEL: <entry1>:
// DISS-NEXT: 1000000: sg
// DISS-LABEL:         b.w {{.*}} <__acle_se_entry1>

// DISS-LABEL: <entry2>:
// DISS-NEXT: 1000008: sg
// DISS-LABEL:         b.w {{.*}} <__acle_se_entry2>

// DISS-LABEL: <__acle_se_entry2>:
// DISS-NEXT:  2000000: nop

// CHECK:    Symbol table '.symtab' contains {{.*}} entries:
// CHECK-NEXT:  Num:  Value  Size Type  Bind   Vis     Ndx Name
// CHECK-NEXT:    0: 00000000   0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK-NEXT:    1: 01000001   8 FUNC  GLOBAL DEFAULT   ABS entry1
// CHECK-NEXT:    2: 01000009   8 FUNC  GLOBAL DEFAULT   ABS entry2

//--- cmse-implib.s
  .include "arm-cmse-macros.s"

  .syntax unified
  .section SECURE1, "ax"

  cmse_veneer entry1, function, global, function, global

  .syntax unified
  .section SECURE2, "ax"

  cmse_veneer entry2, function, global, function, global

//--- cmse-secure-app.s
  .text
  .align  2
  // Main entry point.
  .global secure_entry
  .thumb_func
secure_entry:
  bx lr
  .size  secure_entry, .-secure_entry
