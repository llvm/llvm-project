// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %p/Inputs/plt-aarch64.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld -shared %t.o %t2.so -o %t.so
// RUN: ld.lld %t.o %t2.so -o %t.exe
// RUN: llvm-readobj -S -r %t.so | FileCheck --check-prefix=CHECKDSO %s
// RUN: llvm-objdump -s --section=.got.plt %t.so | FileCheck --check-prefix=DUMPDSO %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.so | FileCheck --check-prefix=DISASMDSO %s
// RUN: llvm-readobj -S -r %t.exe | FileCheck --check-prefix=CHECKEXE %s
// RUN: llvm-objdump -s --section=.got.plt %t.exe | FileCheck --check-prefix=DUMPEXE %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t.exe | FileCheck --check-prefix=DISASMEXE %s

// CHECKDSO:     Name: .plt
// CHECKDSO-NEXT:     Type: SHT_PROGBITS
// CHECKDSO-NEXT:     Flags [
// CHECKDSO-NEXT:       SHF_ALLOC
// CHECKDSO-NEXT:       SHF_EXECINSTR
// CHECKDSO-NEXT:     ]
// CHECKDSO-NEXT:     Address: 0x10340
// CHECKDSO-NEXT:     Offset:
// CHECKDSO-NEXT:     Size: 80
// CHECKDSO-NEXT:     Link:
// CHECKDSO-NEXT:     Info:
// CHECKDSO-NEXT:     AddressAlignment: 16

// CHECKDSO:     Name: .got.plt
// CHECKDSO-NEXT:     Type: SHT_PROGBITS
// CHECKDSO-NEXT:     Flags [
// CHECKDSO-NEXT:       SHF_ALLOC
// CHECKDSO-NEXT:       SHF_WRITE
// CHECKDSO-NEXT:     ]
// CHECKDSO-NEXT:     Address: 0x30450
// CHECKDSO-NEXT:     Offset:
// CHECKDSO-NEXT:     Size: 48
// CHECKDSO-NEXT:     Link:
// CHECKDSO-NEXT:     Info:
// CHECKDSO-NEXT:     AddressAlignment: 8

// CHECKDSO: Relocations [
// CHECKDSO-NEXT:   Section ({{.*}}) .rela.plt {

// &(.got.plt[3]) = 0x30450 + 3 * 8 = 0x30468
// CHECKDSO-NEXT:     0x30468 R_AARCH64_JUMP_SLOT foo

// &(.got.plt[4]) = 0x30450 + 4 * 8 = 0x30470
// CHECKDSO-NEXT:     0x30470 R_AARCH64_JUMP_SLOT bar

// &(.got.plt[5]) = 0x30000 + 5 * 8 = 0x30470
// CHECKDSO-NEXT:     0x30478 R_AARCH64_JUMP_SLOT weak
// CHECKDSO-NEXT:   }
// CHECKDSO-NEXT: ]

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

// CHECKEXE:     Name: .plt
// CHECKEXE-NEXT:     Type: SHT_PROGBITS
// CHECKEXE-NEXT:     Flags [
// CHECKEXE-NEXT:       SHF_ALLOC
// CHECKEXE-NEXT:       SHF_EXECINSTR
// CHECKEXE-NEXT:     ]
// CHECKEXE-NEXT:     Address: 0x2102E0
// CHECKEXE-NEXT:     Offset:
// CHECKEXE-NEXT:     Size: 64
// CHECKEXE-NEXT:     Link:
// CHECKEXE-NEXT:     Info:
// CHECKEXE-NEXT:     AddressAlignment: 16

// CHECKEXE:     Name: .got.plt
// CHECKEXE-NEXT:     Type: SHT_PROGBITS
// CHECKEXE-NEXT:     Flags [
// CHECKEXE-NEXT:       SHF_ALLOC
// CHECKEXE-NEXT:       SHF_WRITE
// CHECKEXE-NEXT:     ]
// CHECKEXE-NEXT:     Address: 0x2303F0
// CHECKEXE-NEXT:     Offset:
// CHECKEXE-NEXT:     Size: 40
// CHECKEXE-NEXT:     Link:
// CHECKEXE-NEXT:     Info:
// CHECKEXE-NEXT:     AddressAlignment: 8

// CHECKEXE: Relocations [
// CHECKEXE-NEXT:   Section ({{.*}}) .rela.plt {

// &(.got.plt[3]) = 0x2303f0 + 3 * 8 = 0x230408
// CHECKEXE-NEXT:     0x230408 R_AARCH64_JUMP_SLOT bar 0x0

// &(.got.plt[4]) = 0x2303f0 + 4 * 8 = 0x230410
// CHECKEXE-NEXT:     0x230410 R_AARCH64_JUMP_SLOT weak 0x0
// CHECKEXE-NEXT:   }
// CHECKEXE-NEXT: ]

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
