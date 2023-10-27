// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t
// RUN: llvm-mc -triple aarch64-none-elf -filetype=obj -o %t/a.o %t/a.s
// RUN: ld.lld --shared %t/a.o -T %t/a.t -o %t/a
// RUN: llvm-objdump --no-show-raw-insn -d --start-address=0x7001004 --stop-address=0x7001010 %t/a | FileCheck %s
// RUN: llvm-objdump --no-show-raw-insn -d --start-address=0x11001008 --stop-address=0x11001014 %t/a | FileCheck --check-prefix=CHECK2 %s
// RUN: rm -f %t/a
/// This test shows that once a long-range thunk has been generated it
/// cannot be written as a short-range thunk. This prevents oscillations
/// in size that can prevent convergence.
/// In pass 0 bl foo requires a long-range thunk to reach foo. The thunk for
/// bar increases the address of foo so that ic can be reaced by bl foo with a
/// a single b instruction.
/// In pass 2 we expect the the long-range thunk to remain long.

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 7001004: adrp    x16, 0x11001000
// CHECK-NEXT:          add     x16, x16, #0x14
// CHECK-NEXT:          br      x16

// CHECK2-LABEL: <__AArch64ADRPThunk_>:
// CHECK2-NEXT: 11001008: adrp    x16, 0x9001000
// CHECK2-NEXT:           add     x16, x16, #0x10
// CHECK2-NEXT:           br      x16


//--- a.t
SECTIONS {
  .foo 0x1000 : { *(.foo.*) }
  .bar 0x11001000 : { *(.bar.*) }
}

//--- a.s
.section .foo.1,"ax",%progbits,unique,1
bl bar

.section .foo.2,"ax",%progbits,unique,1
.space 0x7000000

.section .foo.3,"ax",%progbits,unique,1
.space 0x2000000

.section .foo.4,"ax",%progbits,unique,1
foo:
nop

.section .bar.1,"ax",%progbits,unique,1
nop
nop
.section .bar.2,"ax",%progbits,unique,1
bar:
bl foo
.space 0x8000000
