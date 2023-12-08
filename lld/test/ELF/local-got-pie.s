// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t.o
// RUN: ld.lld --hash-style=sysv %t.o -o %t -pie
// RUN: llvm-readobj -S -d -r %t | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn %t | FileCheck --check-prefix=DISASM %s

.globl _start
_start:
 call foo@gotpcrel

 .hidden foo
 .global foo
foo:
 nop

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// DISASM-NEXT:   1210: callq 0x22d8
// DISASM:      <foo>:
// DISASM-NEXT:   1215: nop

// CHECK:      Name: .got
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT:   SHF_WRITE
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x22D8
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 8

// CHECK:      0x000000006FFFFFF9 RELACOUNT            1

// CHECK:      Relocations [
// CHECK-NEXT:   Section ({{.*}}) .rela.dyn {
// CHECK-NEXT:     0x22D8 R_X86_64_RELATIVE - 0x1215
// CHECK-NEXT:   }
// CHECK-NEXT: ]
