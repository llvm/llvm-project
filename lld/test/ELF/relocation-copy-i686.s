// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=i686-pc-linux %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t.so -o %t.so
// RUN: ld.lld -e main %t.o %t.so -o %t3
// RUN: llvm-readobj -S -r --expand-relocs %t3 | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t3 | FileCheck --check-prefix=CODE %s

.text
.globl main
.align 16, 0x90
.type main,@function
main:
movl $5, x
movl $7, y
movl $9, z

// CHECK:      Name: .bss
// CHECK-NEXT:  Type: SHT_NOBITS
// CHECK-NEXT:  Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address:  0x403270
// CHECK-NEXT:  Offset:
// CHECK-NEXT:  Size: 24
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 16
// CHECK-NEXT:  EntrySize: 0

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rel.dyn {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_386_COPY
// CHECK-NEXT:       Symbol: x
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_386_COPY
// CHECK-NEXT:       Symbol: y
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_386_COPY
// CHECK-NEXT:       Symbol: z
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// 16 is alignment here
// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: <main>:
/// .bss + 0 = 0x403270
// CODE-NEXT: 4011f0:       movl $0x5, 0x403270
/// .bss + 16 = 0x403270 + 16 = 0x403280
// CODE-NEXT: 4011fa:       movl $0x7, 0x403280
/// .bss + 20 = 0x403270 + 20 = 0x403284
// CODE-NEXT: 401204:       movl $0x9, 0x403284
