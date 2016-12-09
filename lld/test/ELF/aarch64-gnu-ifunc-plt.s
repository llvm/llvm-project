// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %S/Inputs/shared2.s -o %t1.o
// RUN: ld.lld %t1.o --shared -o %t.so
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux-gnu %s -o %t.o
// RUN: ld.lld %t.so %t.o -o %tout
// RUN: llvm-objdump -d %tout | FileCheck %s --check-prefix=DISASM
// RUN: llvm-objdump -s %tout | FileCheck %s --check-prefix=GOTPLT
// RUN: llvm-readobj -r -dynamic-table %tout | FileCheck %s
// REQUIRES: aarch64

// Check that the IRELATIVE relocations are after the JUMP_SLOT in the plt
// CHECK: Relocations [
// CHECK-NEXT:   Section (4) .rela.plt {
// CHECK:     0x40018 R_AARCH64_JUMP_SLOT bar2 0x0
// CHECK-NEXT:     0x40020 R_AARCH64_JUMP_SLOT zed2 0x0
// CHECK-NEXT:     0x40028 R_AARCH64_IRELATIVE - 0x20000
// CHECK-NEXT:     0x40030 R_AARCH64_IRELATIVE - 0x20004
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// Check that .got.plt entries point back to PLT header
// GOTPLT: Contents of section .got.plt:
// GOTPLT-NEXT:  40000 00000000 00000000 00000000 00000000
// GOTPLT-NEXT:  40010 00000000 00000000 20000200 00000000
// GOTPLT-NEXT:  40020 20000200 00000000 20000200 00000000
// GOTPLT-NEXT:  40030 20000200 00000000

// Check that the PLTRELSZ tag includes the IRELATIVE relocations
// CHECK: DynamicSection [
// CHECK:   0x0000000000000002 PLTRELSZ             96 (bytes)

// Check that a PLT header is written and the ifunc entries appear last
// DISASM: Disassembly of section .text:
// DISASM-NEXT: foo:
// DISASM-NEXT:    20000:       c0 03 5f d6     ret
// DISASM:      bar:
// DISASM-NEXT:    20004:       c0 03 5f d6     ret
// DISASM:      _start:
// DISASM-NEXT:    20008:       16 00 00 94     bl      #88
// DISASM-NEXT:    2000c:       19 00 00 94     bl      #100
// DISASM-NEXT:    20010:       0c 00 00 94     bl      #48
// DISASM-NEXT:    20014:       0f 00 00 94     bl      #60
// DISASM-NEXT: Disassembly of section .plt:
// DISASM-NEXT: .plt:
// DISASM-NEXT:    20020:       f0 7b bf a9     stp     x16, x30, [sp, #-16]!
// DISASM-NEXT:    20024:       10 01 00 90     adrp    x16, #131072
// DISASM-NEXT:    20028:       11 0a 40 f9     ldr     x17, [x16, #16]
// DISASM-NEXT:    2002c:       10 42 00 91     add     x16, x16, #16
// DISASM-NEXT:    20030:       20 02 1f d6     br      x17
// DISASM-NEXT:    20034:       1f 20 03 d5     nop
// DISASM-NEXT:    20038:       1f 20 03 d5     nop
// DISASM-NEXT:    2003c:       1f 20 03 d5     nop
// DISASM-NEXT:    20040:       10 01 00 90     adrp    x16, #131072
// DISASM-NEXT:    20044:       11 0e 40 f9     ldr     x17, [x16, #24]
// DISASM-NEXT:    20048:       10 62 00 91     add     x16, x16, #24
// DISASM-NEXT:    2004c:       20 02 1f d6     br      x17
// DISASM-NEXT:    20050:       10 01 00 90     adrp    x16, #131072
// DISASM-NEXT:    20054:       11 12 40 f9     ldr     x17, [x16, #32]
// DISASM-NEXT:    20058:       10 82 00 91     add     x16, x16, #32
// DISASM-NEXT:    2005c:       20 02 1f d6     br      x17
// DISASM-NEXT:    20060:       10 01 00 90     adrp    x16, #131072
// DISASM-NEXT:    20064:       11 16 40 f9     ldr     x17, [x16, #40]
// DISASM-NEXT:    20068:       10 a2 00 91     add     x16, x16, #40
// DISASM-NEXT:    2006c:       20 02 1f d6     br      x17
// DISASM-NEXT:    20070:       10 01 00 90     adrp    x16, #131072
// DISASM-NEXT:    20074:       11 1a 40 f9     ldr     x17, [x16, #48]
// DISASM-NEXT:    20078:       10 c2 00 91     add     x16, x16, #48
// DISASM-NEXT:    2007c:       20 02 1f d6     br      x17

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
