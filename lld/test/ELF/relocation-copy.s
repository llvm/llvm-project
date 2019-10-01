// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t.so
// RUN: ld.lld %t.o %t.so -o %t3
// RUN: llvm-readobj -S -r --expand-relocs %t3 | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t3 | FileCheck -check-prefix=CODE %s

.text
.global _start
_start:
movl $5, x
movl $7, y
movl $9, z
movl $x, %edx
movl $y, %edx
movl $z, %edx

// CHECK:      Name: .bss
// CHECK-NEXT:  Type: SHT_NOBITS (0x8)
// CHECK-NEXT:  Flags [ (0x3)
// CHECK-NEXT:   SHF_ALLOC (0x2)
// CHECK-NEXT:   SHF_WRITE (0x1)
// CHECK-NEXT:  ]
// CHECK-NEXT:  Address: 0x203400
// CHECK-NEXT:  Offset:
// CHECK-NEXT:  Size: 24
// CHECK-NEXT:  Link: 0
// CHECK-NEXT:  Info: 0
// CHECK-NEXT:  AddressAlignment: 16
// CHECK-NEXT:  EntrySize: 0

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_X86_64_COPY
// CHECK-NEXT:       Symbol: x
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_X86_64_COPY
// CHECK-NEXT:       Symbol: y
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:     Relocation {
// CHECK-NEXT:       Offset:
// CHECK-NEXT:       Type: R_X86_64_COPY
// CHECK-NEXT:       Symbol: z
// CHECK-NEXT:       Addend: 0x0
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT: ]

// 16 is alignment here
// CODE: Disassembly of section .text:
// CODE-EMPTY:
// CODE-NEXT: _start:
// CODE-NEXT:   movl $0x5, 0x203400
// CODE-NEXT:   movl $0x7, 0x203410
// CODE-NEXT:   movl $0x9, 0x203414
// CODE-NEXT:   movl $0x203400, %edx
// CODE-NEXT:   movl $0x203410, %edx
// CODE-NEXT:   movl $0x203414, %edx
