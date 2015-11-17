// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld2 -shared %t2.o -o %t2.so
// RUN: ld.lld2 %t.o %t2.so -o %t
// RUN: llvm-readobj -s -r %t | FileCheck %s
// RUN: llvm-objdump -d %t | FileCheck --check-prefix=DISASM %s
// REQUIRES: x86

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x120A0
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x120A0 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x120A8 R_X86_64_GLOB_DAT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]


// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:
//  0x120A0 - (0x11000 + 2) - 4 = 4250
//  0x120A0 - (0x11006 + 2) - 4 = 4244
//  0x120A8 - (0x1100c + 2) - 4 = 4246

// DISASM:      _start:
// DISASM-NEXT:   11000:  {{.*}}  jmpq  *4250(%rip)
// DISASM-NEXT:   11006:  {{.*}}  jmpq  *4244(%rip)
// DISASM-NEXT:   1100c:  {{.*}}  jmpq  *4246(%rip)

.global _start
_start:
  jmp *bar@GOTPCREL(%rip)
  jmp *bar@GOTPCREL(%rip)
  jmp *zed@GOTPCREL(%rip)
