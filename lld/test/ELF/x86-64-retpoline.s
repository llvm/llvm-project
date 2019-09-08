// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=x86_64-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared %t2.o -o %t2.so

// RUN: ld.lld -shared %t1.o %t2.so -o %t.exe -z retpolineplt
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 1010:       pushq   8178(%rip)
// CHECK-NEXT:             movq    8179(%rip), %r11
// CHECK-NEXT:             callq   14 <.plt+0x20>
// CHECK-NEXT:             pause
// CHECK-NEXT:             lfence
// CHECK-NEXT:             jmp     -7 <.plt+0x12>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1030:       movq    %r11, (%rsp)
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
// CHECK-NEXT: 1040:       movq    8145(%rip), %r11
// CHECK-NEXT:             callq   -28 <.plt+0x20>
// CHECK-NEXT:             jmp     -47 <.plt+0x12>
// CHECK-NEXT:             pushq   $0
// CHECK-NEXT:             jmp     -75 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT: 1060:       movq    8121(%rip), %r11
// CHECK-NEXT:             callq   -60 <.plt+0x20>
// CHECK-NEXT:             jmp     -79 <.plt+0x12>
// CHECK-NEXT:             pushq   $1
// CHECK-NEXT:             jmp     -107 <.plt>
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3
// CHECK-NEXT:             int3

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 3000 00200000 00000000 00000000 00000000
// CHECK-NEXT: 3010 00000000 00000000 51100000 00000000
// CHECK-NEXT: 3020 71100000 00000000

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
