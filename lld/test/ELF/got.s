// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=t2.so -o %t2.so
// RUN: ld.lld --hash-style=sysv %t.o %t2.so -o %t
// RUN: llvm-readobj -S -r %t | FileCheck %s
// RUN: llvm-objdump --no-print-imm-hex -d %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x202338
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 16
// CHECK-NEXT: Link: 0
// CHECK-NEXT: Info: 0
// CHECK-NEXT: AddressAlignment: 8

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x202338 R_X86_64_GLOB_DAT bar 0x0
// CHECK-NEXT:     0x202340 R_X86_64_GLOB_DAT zed 0x0
// CHECK-NEXT:   }
// CHECK-NEXT: ]


// Unfortunately FileCheck can't do math, so we have to check for explicit
// values:
//  0x202338 - (0x201270 + 2) - 4 = 4290
//  0x202338 - (0x201276 + 2) - 4 = 4284
//  0x202340 - (0x20127c + 2) - 4 = 4286

// DISASM:      <_start>:
// DISASM-NEXT:  201270:  {{.*}}  jmpq  *4290(%rip)
// DISASM-NEXT:  201276:  {{.*}}  jmpq  *4284(%rip)
// DISASM-NEXT:  20127c:  {{.*}}  jmpq  *4286(%rip)

.global _start
_start:
  jmp *bar@GOTPCREL(%rip)
  jmp *bar@GOTPCREL(%rip)
  jmp *zed@GOTPCREL(%rip)
