// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux -position-independent %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux -position-independent %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=t2.so %t2.o -o %t2.so

// RUN: ld.lld %t1.o %t2.so -o %t.exe -z retpolineplt -pie
// RUN: llvm-objdump --no-print-imm-hex -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 32b0 40220000 00000000 00000000 11120000
// CHECK-NEXT: 32c0 31120000

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT: 11d0:       pushl   4(%ebx)
// CHECK-NEXT: 11d6:       pushl   %eax
// CHECK-NEXT: 11d7:       movl    8(%ebx), %eax
// CHECK-NEXT: 11dd:       calll   0x11f0 <.plt+0x20>
// CHECK-NEXT: 11e2:       pause
// CHECK-NEXT: 11e4:       lfence
// CHECK-NEXT: 11e7:       jmp     0x11e2 <.plt+0x12>
// CHECK-NEXT: 11e9:       int3
// CHECK-NEXT: 11ea:       int3
// CHECK-NEXT: 11eb:       int3
// CHECK-NEXT: 11ec:       int3
// CHECK-NEXT: 11ed:       int3
// CHECK-NEXT: 11ee:       int3
// CHECK-NEXT: 11ef:       int3
// CHECK-NEXT: 11f0:       movl    %ecx, (%esp)
// CHECK-NEXT: 11f3:       movl    4(%esp), %ecx
// CHECK-NEXT: 11f7:       movl    %eax, 4(%esp)
// CHECK-NEXT: 11fb:       movl    %ecx, %eax
// CHECK-NEXT: 11fd:       popl    %ecx
// CHECK-NEXT: 11fe:       retl
// CHECK-NEXT: 11ff:       int3
// CHECK-NEXT: 1200:       pushl   %eax
// CHECK-NEXT: 1201:       movl    12(%ebx), %eax
// CHECK-NEXT: 1207:       calll   0x11f0 <.plt+0x20>
// CHECK-NEXT: 120c:       jmp     0x11e2 <.plt+0x12>
// CHECK-NEXT: 1211:       pushl   $0
// CHECK-NEXT: 1216:       jmp     0x11d0 <.plt>
// CHECK-NEXT: 121b:       int3
// CHECK-NEXT: 121c:       int3
// CHECK-NEXT: 121d:       int3
// CHECK-NEXT: 121e:       int3
// CHECK-NEXT: 121f:       int3
// CHECK-NEXT: 1220:       pushl   %eax
// CHECK-NEXT: 1221:       movl    16(%ebx), %eax
// CHECK-NEXT: 1227:       calll   0x11f0 <.plt+0x20>
// CHECK-NEXT: 122c:       jmp     0x11e2 <.plt+0x12>
// CHECK-NEXT: 1231:       pushl   $8
// CHECK-NEXT: 1236:       jmp     0x11d0 <.plt>
// CHECK-NEXT: 123b:       int3
// CHECK-NEXT: 123c:       int3
// CHECK-NEXT: 123d:       int3
// CHECK-NEXT: 123e:       int3
// CHECK-NEXT: 123f:       int3

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
