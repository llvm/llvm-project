// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so

// RUN: ld.lld -shared %t1.o %t2.so -o %t.exe -z retpolineplt -z now
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 23f0 10230000 00000000 00000000 00000000
// CHECK-NEXT: 2400 00000000 00000000 00000000 00000000
// CHECK-NEXT: 2410 00000000 00000000

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 12d0:       callq   11 <.plt+0x10>
// CHECK-NEXT:             pause
// CHECK-NEXT:             lfence
// CHECK-NEXT:             jmp     -7 <.plt+0x5>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 12e0:       movq    %r11, (%rsp)
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
// CHECK-NEXT: 12f0:       movq    4369(%rip), %r11
// CHECK-NEXT:             jmp     -44 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1300:       movq    4361(%rip), %r11
// CHECK-NEXT:             jmp     -60 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
