// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %s -o %t1.o
// RUN: llvm-mc -filetype=obj -triple=i386-unknown-linux %p/Inputs/shared.s -o %t2.o
// RUN: ld.lld -shared -soname=t2.so %t2.o -o %t2.so

// RUN: ld.lld %t1.o %t2.so -o %t.exe -z retpolineplt
// RUN: llvm-objdump -d -s --no-show-raw-insn %t.exe | FileCheck %s

// CHECK:      Disassembly of section .plt:
// CHECK-EMPTY:
// CHECK-NEXT: .plt:
// CHECK-NEXT: 401010:       pushl   4206596
// CHECK-NEXT: 401016:       pushl   %eax
// CHECK-NEXT: 401017:       movl    4206600, %eax
// CHECK-NEXT: 40101c:       calll   15 <.plt+0x20>
// CHECK-NEXT: 401021:       pause
// CHECK-NEXT: 401023:       lfence
// CHECK-NEXT: 401026:       jmp     -7 <.plt+0x11>
// CHECK-NEXT: 401028:       int3
// CHECK-NEXT: 401029:       int3
// CHECK-NEXT: 40102a:       int3
// CHECK-NEXT: 40102b:       int3
// CHECK-NEXT: 40102c:       int3
// CHECK-NEXT: 40102d:       int3
// CHECK-NEXT: 40102e:       int3
// CHECK-NEXT: 40102f:       int3
// CHECK-NEXT: 401030:       movl    %ecx, (%esp)
// CHECK-NEXT: 401033:       movl    4(%esp), %ecx
// CHECK-NEXT: 401037:       movl    %eax, 4(%esp)
// CHECK-NEXT: 40103b:       movl    %ecx, %eax
// CHECK-NEXT: 40103d:       popl    %ecx
// CHECK-NEXT: 40103e:       retl
// CHECK-NEXT: 40103f:       int3
// CHECK-NEXT: 401040:       pushl   %eax
// CHECK-NEXT: 401041:       movl    4206604, %eax
// CHECK-NEXT: 401046:       calll   -27 <.plt+0x20>
// CHECK-NEXT: 40104b:       jmp     -47 <.plt+0x11>
// CHECK-NEXT: 401050:       pushl   $0
// CHECK-NEXT: 401055:       jmp     -74 <.plt>
// CHECK-NEXT: 40105a:       int3
// CHECK-NEXT: 40105b:       int3
// CHECK-NEXT: 40105c:       int3
// CHECK-NEXT: 40105d:       int3
// CHECK-NEXT: 40105e:       int3
// CHECK-NEXT: 40105f:       int3
// CHECK-NEXT: 401060:       pushl   %eax
// CHECK-NEXT: 401061:       movl    4206608, %eax
// CHECK-NEXT: 401066:       calll   -59 <.plt+0x20>
// CHECK-NEXT: 40106b:       jmp     -79 <.plt+0x11>
// CHECK-NEXT: 401070:       pushl   $8
// CHECK-NEXT: 401075:       jmp     -106 <.plt>
// CHECK-NEXT: 40107a:       int3
// CHECK-NEXT: 40107b:       int3
// CHECK-NEXT: 40107c:       int3
// CHECK-NEXT: 40107d:       int3
// CHECK-NEXT: 40107e:       int3
// CHECK-NEXT: 40107f:       int3

// CHECK:      Contents of section .got.plt:
// CHECK-NEXT: 00204000 00000000 00000000 50104000
// CHECK-NEXT: 70104000

.global _start
_start:
  jmp bar@PLT
  jmp zed@PLT
