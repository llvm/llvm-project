// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=t2.so %t2.o -o %t2.so

// RUN: ld.lld %t1.o %t2.so -o %t.exe -z retpolineplt
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 40224000 00000000 00000000 10124000
// CHECK-NEXT: 30124000

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: <.plt>:
// CHECK-NEXT: 4011d0:       pushl   4207276
// CHECK-NEXT: 4011d6:       pushl   %eax
// CHECK-NEXT: 4011d7:       movl    4207280, %eax
// CHECK-NEXT: 4011dc:       calll   0x4011f0 <.plt+0x20>
// CHECK-NEXT: 4011e1:       pause
// CHECK-NEXT: 4011e3:       lfence
// CHECK-NEXT: 4011e6:       jmp     0x4011e1 <.plt+0x11>
// CHECK-NEXT: 4011e8:       int3
// CHECK-NEXT: 4011e9:       int3
// CHECK-NEXT: 4011ea:       int3
// CHECK-NEXT: 4011eb:       int3
// CHECK-NEXT: 4011ec:       int3
// CHECK-NEXT: 4011ed:       int3
// CHECK-NEXT: 4011ee:       int3
// CHECK-NEXT: 4011ef:       int3
// CHECK-NEXT: 4011f0:       movl    %ecx, (%esp)
// CHECK-NEXT: 4011f3:       movl    4(%esp), %ecx
// CHECK-NEXT: 4011f7:       movl    %eax, 4(%esp)
// CHECK-NEXT: 4011fb:       movl    %ecx, %eax
// CHECK-NEXT: 4011fd:       popl    %ecx
// CHECK-NEXT: 4011fe:       retl
// CHECK-NEXT: 4011ff:       int3
// CHECK-NEXT: 401200:       pushl   %eax
// CHECK-NEXT: 401201:       movl    4207284, %eax
// CHECK-NEXT: 401206:       calll   0x4011f0 <.plt+0x20>
// CHECK-NEXT: 40120b:       jmp     0x4011e1 <.plt+0x11>
// CHECK-NEXT: 401210:       pushl   $0
// CHECK-NEXT: 401215:       jmp     0x4011d0 <.plt>
// CHECK-NEXT: 40121a:       int3
// CHECK-NEXT: 40121b:       int3
// CHECK-NEXT: 40121c:       int3
// CHECK-NEXT: 40121d:       int3
// CHECK-NEXT: 40121e:       int3
// CHECK-NEXT: 40121f:       int3
// CHECK-NEXT: 401220:       pushl   %eax
// CHECK-NEXT: 401221:       movl    4207288, %eax
// CHECK-NEXT: 401226:       calll   0x4011f0 <.plt+0x20>
// CHECK-NEXT: 40122b:       jmp     0x4011e1 <.plt+0x11>
// CHECK-NEXT: 401230:       pushl   $8
// CHECK-NEXT: 401235:       jmp     0x4011d0 <.plt>
// CHECK-NEXT: 40123a:       int3
// CHECK-NEXT: 40123b:       int3
// CHECK-NEXT: 40123c:       int3
// CHECK-NEXT: 40123d:       int3
// CHECK-NEXT: 40123e:       int3
// CHECK-NEXT: 40123f:       int3

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
