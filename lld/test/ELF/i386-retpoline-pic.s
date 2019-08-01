// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux -position-independent %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux -position-independent %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=t2.so %t2.o -o %t2.so

// RUN: ld.lld %t1.o %t2.so -o %t.exe -z retpolineplt -pie
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 1010:       pushl   4(%ebx)
// CHECK-NEXT: 1016:       pushl   %eax
// CHECK-NEXT: 1017:       movl    8(%ebx), %eax
// CHECK-NEXT: 101d:       calll   14 <.plt+0x20>
// CHECK-NEXT: 1022:       pause
// CHECK-NEXT: 1024:       lfence
// CHECK-NEXT: 1027:       jmp     -7 <.plt+0x12>
// CHECK-NEXT: 1029:       int3
// CHECK-NEXT: 102a:       int3
// CHECK-NEXT: 102b:       int3
// CHECK-NEXT: 102c:       int3
// CHECK-NEXT: 102d:       int3
// CHECK-NEXT: 102e:       int3
// CHECK-NEXT: 102f:       int3
// CHECK-NEXT: 1030:       movl    %ecx, (%esp)
// CHECK-NEXT: 1033:       movl    4(%esp), %ecx
// CHECK-NEXT: 1037:       movl    %eax, 4(%esp)
// CHECK-NEXT: 103b:       movl    %ecx, %eax
// CHECK-NEXT: 103d:       popl    %ecx
// CHECK-NEXT: 103e:       retl
// CHECK-NEXT: 103f:       int3
// CHECK-NEXT: 1040:       pushl   %eax
// CHECK-NEXT: 1041:       movl    12(%ebx), %eax
// CHECK-NEXT: 1047:       calll   -28 <.plt+0x20>
// CHECK-NEXT: 104c:       jmp     -47 <.plt+0x12>
// CHECK-NEXT: 1051:       pushl   $0
// CHECK-NEXT: 1056:       jmp     -75 <.plt>
// CHECK-NEXT: 105b:       int3
// CHECK-NEXT: 105c:       int3
// CHECK-NEXT: 105d:       int3
// CHECK-NEXT: 105e:       int3
// CHECK-NEXT: 105f:       int3
// CHECK-NEXT: 1060:       pushl   %eax
// CHECK-NEXT: 1061:       movl    16(%ebx), %eax
// CHECK-NEXT: 1067:       calll   -60 <.plt+0x20>
// CHECK-NEXT: 106c:       jmp     -79 <.plt+0x12>
// CHECK-NEXT: 1071:       pushl   $8
// CHECK-NEXT: 1076:       jmp     -107 <.plt>
// CHECK-NEXT: 107b:       int3
// CHECK-NEXT: 107c:       int3
// CHECK-NEXT: 107d:       int3
// CHECK-NEXT: 107e:       int3
// CHECK-NEXT: 107f:       int3

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 3000 00200000 00000000 00000000 51100000
// CHECK-NEXT: 3010 71100000

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
