// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %p/Inputs/plt-aarch64.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld -shared %t.o %t2.so -o %t.so
// RUN: ld.lld %t.o %t2.so -o %t.exe
// RUN: llvm-readelf -S -r %t.so | FileCheck --check-prefix=CHECKDSO %s
// RUN: llvm-objdump -s --section=.got.plt %t.so | FileCheck --check-prefix=DUMPDSO %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck --check-prefix=DISASMDSO %s
// RUN: llvm-readelf -S -r %t.exe | FileCheck --check-prefix=CHECKEXE %s
// RUN: llvm-objdump -s --section=.got.plt %t.exe | FileCheck --check-prefix=DUMPEXE %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.exe | FileCheck --check-prefix=DISASMEXE %s

// CHECKDSO-LABEL: Section Headers:
///          Name     Type     Address          Off    Size   ES Flg Lk Inf Al
// CHECKDSO: .plt     PROGBITS 0000000000010340 000340 000050 00 AXy  0   0 16
// CHECKDSO: .got.plt PROGBITS 0000000000030450 000450 000030 00  WA  0   0  8

// CHECKDSO-LABEL: Relocation section '.rela.plt' at offset 0x2e8 contains 3 entries:
// CHECKDSO-NEXT:      Offset             Info             Type            Symbol's Value  Symbol's Name + Addend
/// &(.got.plt[3]) = 0x30450 + 3 * 8 = 0x30468
// CHECKDSO-NEXT:  0000000000030468  0000000400000402 R_AARCH64_JUMP_SLOT 000000000001033c foo + 0
/// &(.got.plt[4]) = 0x30450 + 4 * 8 = 0x30470
// CHECKDSO-NEXT:  0000000000030470  0000000100000402 R_AARCH64_JUMP_SLOT 0000000000000000 bar + 0
/// &(.got.plt[5]) = 0x30000 + 5 * 8 = 0x30478
// CHECKDSO-NEXT:  0000000000030478  0000000200000402 R_AARCH64_JUMP_SLOT 0000000000000000 weak + 0

// DUMPDSO: Contents of section .got.plt:
// .got.plt[0..2] = 0 (reserved)
// .got.plt[3..5] = .plt = 0x10010
// DUMPDSO-NEXT: 30450 00000000 00000000 00000000 00000000
// DUMPDSO-NEXT: 30460 00000000 00000000 40030100 00000000
// DUMPDSO-NEXT: 30470 40030100 00000000 40030100 00000000

// DISASMDSO: <_start>:
// DISASMDSO-NEXT:     10330: b       0x10360 <foo@plt>
// DISASMDSO-NEXT:     10334: b       0x10370 <bar@plt>
// DISASMDSO-NEXT:     10338: b       0x10380 <weak@plt>

// DISASMDSO: <foo>:
// DISASMDSO-NEXT:     1033c: nop

// DISASMDSO: Disassembly of section .plt:
// DISASMDSO-EMPTY:
// DISASMDSO-NEXT: <.plt>:
// DISASMDSO-NEXT:     10340: stp     x16, x30, [sp, #-0x10]!
// &(.got.plt[2]) = 0x30450 + 2 * 8 = 0x30460
// DISASMDSO-NEXT:     10344: adrp    x16, 0x30000
// DISASMDSO-NEXT:     10348: ldr     x17, [x16, #0x460]
// DISASMDSO-NEXT:     1034c: add     x16, x16, #0x460
// DISASMDSO-NEXT:     10350: br      x17
// DISASMDSO-NEXT:     10354: nop
// DISASMDSO-NEXT:     10358: nop
// DISASMDSO-NEXT:     1035c: nop

// foo@plt 0x30468
// &.got.plt[foo] = 0x30468
// DISASMDSO-EMPTY:
// DISASMDSO-NEXT:   <foo@plt>:
// DISASMDSO-NEXT:     10360: adrp    x16, 0x30000
// DISASMDSO-NEXT:     10364: ldr     x17, [x16, #0x468]
// DISASMDSO-NEXT:     10368: add     x16, x16, #0x468
// DISASMDSO-NEXT:     1036c: br      x17

// bar@plt
// &.got.plt[foo] = 0x30470
// DISASMDSO-EMPTY:
// DISASMDSO-NEXT:   <bar@plt>:
// DISASMDSO-NEXT:     10370: adrp    x16, 0x30000
// DISASMDSO-NEXT:     10374: ldr     x17, [x16, #0x470]
// DISASMDSO-NEXT:     10378: add     x16, x16, #0x470
// DISASMDSO-NEXT:     1037c: br      x17

// weak@plt
// 0x30468 = 0x10000 + 131072 + 1128
// DISASMDSO-EMPTY:
// DISASMDSO-NEXT:   <weak@plt>:
// DISASMDSO-NEXT:     10380: adrp    x16, 0x30000
// DISASMDSO-NEXT:     10384: ldr     x17, [x16, #0x478]
// DISASMDSO-NEXT:     10388: add     x16, x16, #0x478
// DISASMDSO-NEXT:     1038c: br      x17

// CHECKEXE-LABEL: Section Headers:
///          Name     Type     Address          Off    Size   ES Flg Lk Inf Al
// CHECKEXE: .plt     PROGBITS 00000000002102e0 0002e0 000040 00 AXy  0   0 16
// CHECKEXE: .got.plt PROGBITS 00000000002303f0 0003f0 000028 00  WA  0   0  8

// CHECKEXE-LABEL: Relocation section '.rela.plt' at offset 0x298 contains 2 entries:
// CHECKEXE-NEXT:      Offset             Info             Type               Symbol's Value  Symbol's Name + Addend
/// &(.got.plt[3]) = 0x2303f0 + 3 * 8 = 0x230408
// CHECKEXE-NEXT:  0000000000230408  0000000100000402 R_AARCH64_JUMP_SLOT    0000000000000000 bar + 0
/// &(.got.plt[4]) = 0x2303f0 + 4 * 8 = 0x230410
// CHECKEXE-NEXT:  0000000000230410  0000000200000402 R_AARCH64_JUMP_SLOT    0000000000000000 weak + 0

// DUMPEXE: Contents of section .got.plt:
// .got.plt[0..2] = 0 (reserved)
// .got.plt[3..4] = .plt = 0x40010
// DUMPEXE-NEXT:  2303f0 00000000 00000000 00000000 00000000
// DUMPEXE-NEXT:  230400 00000000 00000000 e0022100 00000000
// DUMPEXE-NEXT:  230410 e0022100 00000000

// DISASMEXE: <_start>:
// DISASMEXE-NEXT:    2102c8: b 0x2102d4 <foo>
// DISASMEXE-NEXT:    2102cc: b 0x210300 <bar@plt>
// DISASMEXE-NEXT:    2102d0: b 0x210310 <weak@plt>

// DISASMEXE: <foo>:
// DISASMEXE-NEXT:    2102d4: nop

// DISASMEXE: Disassembly of section .plt:
// DISASMEXE-EMPTY:
// DISASMEXE-NEXT: <.plt>:
// DISASMEXE-NEXT:    2102e0: stp     x16, x30, [sp, #-0x10]!
// &(.got.plt[2]) = 0x2303f0 + 2 * 8 = 0x230400
// DISASMEXE-NEXT:    2102e4: adrp    x16, 0x230000
// DISASMEXE-NEXT:    2102e8: ldr     x17, [x16, #0x400]
// DISASMEXE-NEXT:    2102ec: add     x16, x16, #0x400
// DISASMEXE-NEXT:    2102f0: br      x17
// DISASMEXE-NEXT:    2102f4: nop
// DISASMEXE-NEXT:    2102f8: nop
// DISASMEXE-NEXT:    2102fc: nop

// bar@plt
// DISASMEXE-EMPTY:
// DISASMEXE-NEXT:   <bar@plt>:
// DISASMEXE-NEXT:    210300: adrp    x16, 0x230000
// DISASMEXE-NEXT:    210304: ldr     x17, [x16, #0x408]
// DISASMEXE-NEXT:    210308: add     x16, x16, #0x408
// DISASMEXE-NEXT:    21030c: br      x17

// weak@plt
// DISASMEXE-EMPTY:
// DISASMEXE-NEXT:   <weak@plt>:
// DISASMEXE-NEXT:    210310: adrp    x16, 0x230000
// DISASMEXE-NEXT:    210314: ldr     x17, [x16, #0x410]
// DISASMEXE-NEXT:    210318: add     x16, x16, #0x410
// DISASMEXE-NEXT:    21031c: br      x17

.global _start,foo,bar
.weak weak
_start:
  b foo
  b bar
  b weak

.section .text2,"ax",@progbits
foo:
  nop
