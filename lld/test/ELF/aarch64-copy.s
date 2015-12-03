// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=aarch64-pc-freebsd %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so
// RUN: ld.lld %t.o %t2.so -o %t3
// RUN: llvm-readobj -s -r --expand-relocs -symbols %t3 | FileCheck %s
// RUN: llvm-objdump -d %t3 | FileCheck -check-prefix=CODE %s
// RUN: llvm-objdump -s -section=.data %t3 | FileCheck -check-prefix=DATA %s

.text
.globl _start
_start:
    adr x1, x
    adrp x2, y
    add x2, x2, :lo12:y
.data
    .word z

// CHECK:     Name: .bss
// CHECK-NEXT:     Type: SHT_NOBITS
// CHECK-NEXT:     Flags [
// CHECK-NEXT:       SHF_ALLOC
// CHECK-NEXT:       SHF_WRITE
// CHECK-NEXT:     ]
// CHECK-NEXT:     Address: 0x120B0
// CHECK-NEXT:     Offset:
// CHECK-NEXT:     Size: 24
// CHECK-NEXT:     Link:
// CHECK-NEXT:     Info:
// CHECK-NEXT:     AddressAlignment: 16

// CHECK: Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x120B0
// CHECK-NEXT:       Type: R_AARCH64_COPY
// CHECK-NEXT:       Symbol: x
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x120C0
// CHECK-NEXT:       Type: R_AARCH64_COPY
// CHECK-NEXT:       Symbol: y
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset: 0x120C4
// CHECK-NEXT:       Type: R_AARCH64_COPY
// CHECK-NEXT:       Symbol: z
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// CHECK: Symbols [
// CHECK:     Name: x
// CHECK-NEXT:     Value: 0x120B0
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK:     Name: y
// CHECK-NEXT:     Value: 0x120C0
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK:     Name: z
// CHECK-NEXT:     Value: 0x120C4
// CHECK-NEXT:     Size: 4
// CHECK-NEXT:     Binding: Global
// CHECK-NEXT:     Type: Object
// CHECK-NEXT:     Other:
// CHECK-NEXT:     Section: .bss
// CHECK: ]

// CODE: Disassembly of section .text:
// CODE-NEXT: _start:
// S(x) = 0x120B0, A = 0, P = 0x11000
// S + A - P = 0x10B0 = 4272
// CODE-NEXT:  11000: {{.*}} adr  x1, #4272
// S(y) = 0x120C0, A = 0, P = 0x11004
// Page(S + A) - Page(P) = 0x12000 - 0x11000 = 0x1000 - 4096
// CODE-NEXT:  11004: {{.*}} adrp x2, #4096
// S(y) = 0x120C0, A = 0
// (S + A) & 0xFFF = 0xC0 = 192
// CODE-NEXT:  11008: {{.*}} add  x2, x2, #192

// DATA: Contents of section .data:
// S(z) = 0x120c4
// DATA-NEXT:  120a0 c4200100
