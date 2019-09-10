// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -soname=so -o %t2.so

// RUN: ld.lld -shared %t1.o %t2.so -o %t.exe -z retpolineplt -z now
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 1010:       callq   11 <.plt+0x10>
// CHECK-NEXT:             pause
// CHECK-NEXT:             lfence
// CHECK-NEXT:             jmp     -7 <.plt+0x5>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1020:       movq    %r11, (%rsp)
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
// CHECK-NEXT: 1030:       movq    4289(%rip), %r11
// CHECK-NEXT:             jmp     -44 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1040:       movq    4281(%rip), %r11
// CHECK-NEXT:             jmp     -60 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 20e0 00200000 00000000 00000000 00000000
// CHECK-NEXT: 20f0 00000000 00000000 00000000 00000000
// CHECK-NEXT: 2100 00000000 00000000

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
