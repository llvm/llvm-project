// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname fixed-length-string.so -o %t2.so
// RUN: ld.lld --no-relax %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r --symbols %t | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t | FileCheck --check-prefix=CODE %s
// RUN: llvm-objdump -s --section=.rodata %t | FileCheck --check-prefix=RODATA %s

.text
.globl _start
_start:
    adr x1, x
    adrp x2, y
    add x2, x2, :lo12:y
.rodata
    .word z

// CHECK:     Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x2303F0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 16

// CHECK: Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x2303F0 R_AARCH64_COPY x 0x0
// CHECK-NEXT:     0x230400 R_AARCH64_COPY y 0x0
// CHECK-NEXT:     0x230404 R_AARCH64_COPY z 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK: Symbols [
// CHECK:     Name: x
// CHECK-NEXT:     Value: 0x2303F0
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK:     Name: y
// CHECK-NEXT:     Value: 0x230400
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK:     Name: z
// CHECK-NEXT:     Value: 0x230404
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK: ]

// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: <_start>:
// S + A - P = 0x2303f0 + 0 - 0x21031c = 131284
// CODE-NEXT:  21031c: adr  x1, 0x2303f0
// Page(S + A) - Page(P) = Page(0x230400) - Page(0x210320) = 131072
// CODE-NEXT:  210320: adrp x2, 0x230000
// (S + A) & 0xFFF = (0x230400 + 0) & 0xFFF = 1024
// CODE-NEXT:  210324: add  x2, x2, #1024

// RODATA: Contents of section .rodata:
// S(z) = 0x230404
// RODATA-NEXT:  200318 04042300
