// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/relocation-copy.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t.so
// RUN: ld.lld %t.o %t.so -o %t3
// RUN: llvm-readobj -S -r --expand-relocs %t3 | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex %t3 | FileCheck --check-prefix=CODE %s

/// In a PIE, the address of a copy relocation is not a link-time constant.
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux --defsym PIE=1 %s -o %t-pie.o
// RUN: not ld.lld -pie %t-pie.o %t.so -o /dev/null 2>&1 | FileCheck --check-prefix=PIE --implicit-check-not=error: %s
// PIE-COUNT-3: error: relocation R_X86_64_32S cannot be used against symbol '{{[xyz]}}'; recompile with -fPIC
// PIE-COUNT-3: error: relocation R_X86_64_32 cannot be used against symbol '{{[xyz]}}'; recompile with -fPIC
// PIE:         error: relocation R_X86_64_64 cannot be used against symbol 'x'; recompile with -fPIC

.text
.global _start
_start:
movl $5, x
movl $7, y
movl $9, z
movl $x, %edx
movl $y, %edx
movl $z, %edx

.ifdef PIE
.section .rodata,"a"
.quad x
/// st_size is a link-time constant.
.quad x@SIZE
.endif

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
// CODE-NEXT: <_start>:
// CODE-NEXT:   movl $0x5, 0x203400
// CODE-NEXT:   movl $0x7, 0x203410
// CODE-NEXT:   movl $0x9, 0x203414
// CODE-NEXT:   movl $0x203400, %edx
// CODE-NEXT:   movl $0x203410, %edx
// CODE-NEXT:   movl $0x203414, %edx
