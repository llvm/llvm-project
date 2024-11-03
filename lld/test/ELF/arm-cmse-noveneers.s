// REQUIRES: arm
/// Test that addresses of existing secure gateway veneers are output in the CMSE import library.
/// Test that .gnu.sgstubs is size 0 when no linker synthesized secure gateway veneers are created.

// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.main %s -I %S/Inputs -o %t.o
// RUN: ld.lld --cmse-implib -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 %t.o -o %t --out-implib=%t.lib
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck %s
// RUN: llvm-readelf -S %t | FileCheck %s --check-prefixes=SGSTUBSSIZE
// RUN: llvm-readelf -s %t.lib | FileCheck %s --check-prefixes=IMPLIBSYMS

// CHECK: Disassembly of section .text:

// CHECK-LABEL: <existing_veneer>:
// CHECK-NEXT:     8000: sg
// CHECK-NEXT:     8004: nop

// CHECK-LABEL: <__acle_se_existing_veneer>:
// CHECK-NEXT:     8006: nop

 .include "arm-cmse-macros.s"

  cmse_no_veneer existing_veneer, function, global, function, global

///                      Name          Type         Address    Off   Size ES Flg Lk Inf Al
// SGSTUBSSIZE: .gnu.sgstubs      PROGBITS        00020000 020000 000000 08  AX  0   0 32

// IMPLIBSYMS:      Symbol table '.symtab' contains 2 entries:
// IMPLIBSYMS-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// IMPLIBSYMS-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// IMPLIBSYMS-NEXT:      1: 00008001     8 FUNC    GLOBAL DEFAULT   ABS existing_veneer
