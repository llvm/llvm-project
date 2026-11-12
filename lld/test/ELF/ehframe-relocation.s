// REQUIRES: x86
// RUN: llvm-mc -filetype=obj -triple=x86_64-pc-linux %s -o %t.o
// RUN: echo '.cfi_startproc; .cfi_endproc' | llvm-mc -filetype=obj -triple=x86_64 - -o %t2.o
// RUN: ld.lld %t.o %t2.o -o %t
// RUN: llvm-readobj -S %t | FileCheck %s
// RUN: llvm-objdump -d --print-imm-hex %t | FileCheck --check-prefix=DISASM %s

// CHECK:      Name: .eh_frame
// CHECK-NEXT: Type: SHT_PROGBITS
// CHECK-NEXT: Flags [
// CHECK-NEXT:   SHF_ALLOC
// CHECK-NEXT: ]
// CHECK-NEXT: Address: 0x200120
// CHECK-NEXT: Offset:
// CHECK-NEXT: Size: 52
// CHECK-NOT: .eh_frame

// DISASM:      Disassembly of section .text:
// DISASM-EMPTY:
// DISASM-NEXT: <_start>:
// DISASM-NEXT:   movq 0x200120, %rax
// DISASM-NEXT:   leaq {{.*}}(%rip), %rax  # {{.*}} <__EH_FRAME_LIST__>

.section .eh_frame,"a",@unwind
__EH_FRAME_LIST__:

.section .text
.globl _start
_start:
 movq .eh_frame, %rax  # addend=0
 leaq __EH_FRAME_LIST__(%rip), %rax  # addend=-4, used by libclang_rt.crtbegin.o
