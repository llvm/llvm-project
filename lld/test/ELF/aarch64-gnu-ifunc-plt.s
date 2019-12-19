// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %S/Inputs/shared2.s -o %t1.o
// RUN: ld.lld %t1.o --shared --soname=t.so -o %t.so
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.so %t.o -o %tout
// RUN: llvm-objdump -d --no-show-raw-insn %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r --dynamic-table %tout | FileCheck %s

// Check that the IRELATIVE relocations are after the JUMP_SLOT in the plt
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rela.dyn {
// CHECK-NEXT:     0x230468 R_AARCH64_IRELATIVE - 0x2102D8
// CHECK-NEXT:     0x230470 R_AARCH64_IRELATIVE - 0x2102DC
// CHECK-NEXT:   }
// CHECK-NEXT:   Section (5) .rela.plt {
// CHECK-NEXT:     0x230458 R_AARCH64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x230460 R_AARCH64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Check that .got.plt entries point back to PLT header
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  230440 00000000 00000000 00000000 00000000
// GOTPLT-NEXT:  230450 00000000 00000000 f0022100 00000000
// GOTPLT-NEXT:  230460 f0022100 00000000 f0022100 00000000
// GOTPLT-NEXT:  230470 f0022100 00000000

// Check that the PLTRELSZ tag does not include the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:   0x0000000000000008 RELASZ               48 (bytes)
// CHECK:   0x0000000000000002 PLTRELSZ             48 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: foo:
// DISASM-NEXT:    2102d8: ret
// DISASM:      bar:
// DISASM-NEXT:    2102dc: ret
// DISASM:      _start:
// DISASM-NEXT:    2102e0: bl      #80 <zed2+0x210330>
// DISASM-NEXT:    2102e4: bl      #92 <zed2+0x210340>
// DISASM-NEXT:    2102e8: bl      #40 <bar2@plt>
// DISASM-NEXT:    2102ec: bl      #52 <zed2@plt>
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-EMPTY:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    2102f0: stp     x16, x30, [sp, #-16]!
// DISASM-NEXT:    2102f4: adrp    x16, #131072
// DISASM-NEXT:    2102f8: ldr     x17, [x16, #1104]
// DISASM-NEXT:    2102fc: add     x16, x16, #1104
// DISASM-NEXT:    210300: br      x17
// DISASM-NEXT:    210304: nop
// DISASM-NEXT:    210308: nop
// DISASM-NEXT:    21030c: nop
// DISASM-EMPTY:
// DISASM-NEXT:   bar2@plt:
// DISASM-NEXT:    210310: adrp    x16, #131072
// DISASM-NEXT:    210314: ldr     x17, [x16, #1112]
// DISASM-NEXT:    210318: add     x16, x16, #1112
// DISASM-NEXT:    21031c: br      x17
// DISASM-EMPTY:
// DISASM-NEXT:   zed2@plt:
// DISASM-NEXT:    210320: adrp    x16, #131072
// DISASM-NEXT:    210324: ldr     x17, [x16, #1120]
// DISASM-NEXT:    210328: add     x16, x16, #1120
// DISASM-NEXT:    21032c: br      x17
// DISASM-EMPTY:
// DISASM-NEXT: Disassembly of section .iplt:
// DISASM-EMPTY:
// DISASM-NEXT: .iplt:
// DISASM-NEXT:    210330: adrp    x16, #131072
// DISASM-NEXT:    210334: ldr     x17, [x16, #1128]
// DISASM-NEXT:    210338: add     x16, x16, #1128
// DISASM-NEXT:    21033c: br      x17
// DISASM-NEXT:    210340: adrp    x16, #131072
// DISASM-NEXT:    210344: ldr     x17, [x16, #1136]
// DISASM-NEXT:    210348: add     x16, x16, #1136
// DISASM-NEXT:    21034c: br      x17

.text
.type foo STT_GNU_IFUNC
.globl foo
foo:
 ret

.type bar STT_GNU_IFUNC
.globl bar
bar:
 ret

.globl _start
_start:
 bl foo
 bl bar
 bl bar2
 bl zed2
