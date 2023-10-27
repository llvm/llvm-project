// REQUIRES: arm
/// Test that addresses of secure gateways in an old import library are maintained in new import libraries.

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.base app -o app.o
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.base implib-v1 -I %S/Inputs -o 1.o
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj --triple=thumbv8m.base implib-v2 -I %S/Inputs -o 2.o

// RUN: ld.lld -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o 1 app.o --out-implib=1.lib --cmse-implib 1.o
// RUN: llvm-readelf -s 1 1.lib | FileCheck %s --check-prefixes=CHECK1

// RUN: ld.lld -Ttext=0x8000 --section-start .gnu.sgstubs=0x20000 -o 2 app.o --out-implib=2.lib --in-implib=1.lib --cmse-implib 2.o
// RUN: llvm-readelf -s 2 2.lib | FileCheck %s --check-prefixes=CHECK2

//--- app

	.align	2
	.global	secure_entry
	.type	secure_entry, %function
secure_entry:
	nop
	.size	secure_entry, .-secure_entry

//--- implib-v1

    .include "arm-cmse-macros.s"

    .syntax unified
    .text

  cmse_veneer foo, function, global, function, global
  cmse_veneer bar, function, weak, function, global
  cmse_no_veneer no_veneer1, function, weak, function, global
  cmse_no_veneer no_veneer2, function, weak, function, weak

//--- implib-v2

    .include "arm-cmse-macros.s"

    .syntax unified
    .text

  cmse_veneer baz, function, weak, function, global
  cmse_veneer foo, function, global, function, global
  cmse_veneer bar, function, weak, function, global
  cmse_veneer qux, function, global, function, global
  cmse_no_veneer no_veneer1, function, weak, function, global
  cmse_no_veneer no_veneer2, function, weak, function, weak

/// Executable 1
// CHECK1:      File:
// CHECK1:      Symbol table '.symtab' contains 13 entries:
// CHECK1-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// CHECK1-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK1-NEXT:      1: 00020000     0 NOTYPE  LOCAL  DEFAULT     2 $t
// CHECK1-NEXT:      2: 00008000     0 NOTYPE  LOCAL  DEFAULT     1 $t.0
// CHECK1-NEXT:      3: 00008004     0 NOTYPE  LOCAL  DEFAULT     1 $t.0
// CHECK1-NEXT:      4: 00008001     2 FUNC    GLOBAL DEFAULT     1 secure_entry
// CHECK1-NEXT:      5: 00020001     8 FUNC    GLOBAL DEFAULT     2 foo
// CHECK1-NEXT:      6: 00008005     2 FUNC    GLOBAL DEFAULT     1 __acle_se_foo
// CHECK1-NEXT:      7: 00020009     8 FUNC    WEAK   DEFAULT     2 bar
// CHECK1-NEXT:      8: 00008009     2 FUNC    GLOBAL DEFAULT     1 __acle_se_bar
// CHECK1-NEXT:      9: 0000800d     8 FUNC    WEAK   DEFAULT     1 no_veneer1
// CHECK1-NEXT:     10: 00008013     2 FUNC    GLOBAL DEFAULT     1 __acle_se_no_veneer1
// CHECK1-NEXT:     11: 00008015     8 FUNC    WEAK   DEFAULT     1 no_veneer2
// CHECK1-NEXT:     12: 0000801b     2 FUNC    WEAK   DEFAULT     1 __acle_se_no_veneer2


/// Import library 1
// CHECK1:      File:
// CHECK1:      Symbol table '.symtab' contains 5 entries:
// CHECK1-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// CHECK1-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK1-NEXT:      1: 0000800d     8 FUNC    WEAK   DEFAULT   ABS no_veneer1
// CHECK1-NEXT:      2: 00008015     8 FUNC    WEAK   DEFAULT   ABS no_veneer2
// CHECK1-NEXT:      3: 00020001     8 FUNC    GLOBAL DEFAULT   ABS foo
// CHECK1-NEXT:      4: 00020009     8 FUNC    WEAK   DEFAULT   ABS bar

/// Executable 2
// CHECK2:      File:
// CHECK2:      Symbol table '.symtab' contains 17 entries:
// CHECK2-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// CHECK2-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK2-NEXT:      1: 00020000     0 NOTYPE  LOCAL  DEFAULT     2 $t
// CHECK2-NEXT:      2: 00008000     0 NOTYPE  LOCAL  DEFAULT     1 $t.0
// CHECK2-NEXT:      3: 00008004     0 NOTYPE  LOCAL  DEFAULT     1 $t.0
// CHECK2-NEXT:      4: 00008001     2 FUNC    GLOBAL DEFAULT     1 secure_entry
// CHECK2-NEXT:      5: 00020011     8 FUNC    WEAK   DEFAULT     2 baz
// CHECK2-NEXT:      6: 00008005     2 FUNC    GLOBAL DEFAULT     1 __acle_se_baz
// CHECK2-NEXT:      7: 00020001     8 FUNC    GLOBAL DEFAULT     2 foo
// CHECK2-NEXT:      8: 00008009     2 FUNC    GLOBAL DEFAULT     1 __acle_se_foo
// CHECK2-NEXT:      9: 00020009     8 FUNC    WEAK   DEFAULT     2 bar
// CHECK2-NEXT:     10: 0000800d     2 FUNC    GLOBAL DEFAULT     1 __acle_se_bar
// CHECK2-NEXT:     11: 00020019     8 FUNC    GLOBAL DEFAULT     2 qux
// CHECK2-NEXT:     12: 00008011     2 FUNC    GLOBAL DEFAULT     1 __acle_se_qux
// CHECK2-NEXT:     13: 00008015     8 FUNC    WEAK   DEFAULT     1 no_veneer1
// CHECK2-NEXT:     14: 0000801b     2 FUNC    GLOBAL DEFAULT     1 __acle_se_no_veneer1
// CHECK2-NEXT:     15: 0000801d     8 FUNC    WEAK   DEFAULT     1 no_veneer2
// CHECK2-NEXT:     16: 00008023     2 FUNC    WEAK   DEFAULT     1 __acle_se_no_veneer2


/// Note that foo retains its address from Import library 1 (0x000020001)
/// New entry functions, baz and qux, use addresses not used by Import library 1.
/// Import library 2
// CHECK2:      File:
// CHECK2:      Symbol table '.symtab' contains 7 entries:
// CHECK2-NEXT:    Num:    Value  Size Type    Bind   Vis       Ndx Name
// CHECK2-NEXT:      0: 00000000     0 NOTYPE  LOCAL  DEFAULT   UND
// CHECK2-NEXT:      1: 00008015     8 FUNC    WEAK   DEFAULT   ABS no_veneer1
// CHECK2-NEXT:      2: 0000801d     8 FUNC    WEAK   DEFAULT   ABS no_veneer2
// CHECK2-NEXT:      3: 00020001     8 FUNC    GLOBAL DEFAULT   ABS foo
// CHECK2-NEXT:      4: 00020009     8 FUNC    WEAK   DEFAULT   ABS bar
// CHECK2-NEXT:      5: 00020011     8 FUNC    WEAK   DEFAULT   ABS baz
// CHECK2-NEXT:      6: 00020019     8 FUNC    GLOBAL DEFAULT   ABS qux
