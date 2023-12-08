// REQUIRES: x86

// Reserve space for copy relocations of read-only symbols in .bss.rel.ro

// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %p/Inputs/relocation-copy-relro.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t.so
// RUN: ld.lld %t.o %t.so -o %t3
// RUN: llvm-readobj -S -l -r %t3 | FileCheck %s

/// Due to -z rodynamic, The only RELRO section is .bss.rel.ro. Test that we
/// still append the .relro_padding section.
// RUN: ld.lld -z rodynamic %t.o %t.so -o %t4
// RUN: llvm-readelf -S -l %t4 | FileCheck %s --check-prefix=CHECK2

// CHECK:        Name: .bss.rel.ro
// CHECK-NEXT:   Type: SHT_NOBITS (0x8)
// CHECK-NEXT:   Flags [ (0x3)
// CHECK-NEXT:     SHF_ALLOC (0x2)
// CHECK-NEXT:     SHF_WRITE (0x1)
// CHECK-NEXT:   ]
// CHECK-NEXT:   Address: 0x202368
// CHECK-NEXT:   Offset: 0x368
// CHECK-NEXT:   Size: 8

// CHECK:      Type: PT_GNU_RELRO (0x6474E552)
// CHECK-NEXT: Offset: 0x2A8
// CHECK-NEXT: VirtualAddress: 0x2022A8
// CHECK-NEXT: PhysicalAddress: 0x2022A8
// CHECK-NEXT: FileSize: 192
// CHECK-NEXT: MemSize: 3416

// CHECK: 0x202368 R_X86_64_COPY a 0x0
// CHECK: 0x20236C R_X86_64_COPY b 0x0

// CHECK2:      LOAD           0x000356 0x0000000000202356 0x0000000000202356 0x000000 0x000caa RW  0x1000
// CHECK2:      DYNAMIC        0x000258 0x0000000000200258 0x0000000000200258 0x0000b0 0x0000b0 R   0x8
// CHECK2:      GNU_RELRO      0x000356 0x0000000000202356 0x0000000000202356 0x000000 0x000caa R   0x1

// CHECK2:      Section to Segment mapping:
// CHECK2-NEXT:  Segment Sections...
// CHECK2-NEXT:   00
// CHECK2-NEXT:   01     .dynsym .gnu.hash .hash .dynamic .dynstr .rela.dyn
// CHECK2-NEXT:   02     .text
// CHECK2-NEXT:   03     .bss.rel.ro .relro_padding
// CHECK2-NEXT:   04     .dynamic
// CHECK2-NEXT:   05     .bss.rel.ro .relro_padding
// CHECK2-NEXT:   06

.text
.global _start
_start:
movl $1, a
movl $2, b
