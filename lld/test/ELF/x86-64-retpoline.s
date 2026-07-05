// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so

// RUN: ld.lld -shared %t1.o %t2.so -o %t.exe -z retpolineplt
// RUN: llvm-objdump --no-print-imm-hex -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 3430 70230000 00000000 00000000 00000000
// CHECK-NEXT: 3440 00000000 00000000 41130000 00000000
// CHECK-NEXT: 3450 61130000 00000000

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT: 1300:       pushq   8498(%rip)
// CHECK-NEXT:             movq    8499(%rip), %r11
// CHECK-NEXT:             callq   0x1320 <.plt+0x20>
// CHECK-NEXT:             pause
// CHECK-NEXT:             lfence
// CHECK-NEXT:             jmp     0x1312 <.plt+0x12>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1320:       movq    %r11, (%rsp)
// CHECK-NEXT:             retq
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1330:       movq    8465(%rip), %r11
// CHECK-NEXT:             callq   0x1320 <.plt+0x20>
// CHECK-NEXT:             jmp     0x1312 <.plt+0x12>
// CHECK-NEXT:             pushq   $0
// CHECK-NEXT:             jmp     0x1300 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1350:       movq    8441(%rip), %r11
// CHECK-NEXT:             callq   0x1320 <.plt+0x20>
// CHECK-NEXT:             jmp     0x1312 <.plt+0x12>
// CHECK-NEXT:             pushq   $1
// CHECK-NEXT:             jmp     0x1300 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
