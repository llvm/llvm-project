// REQUIRES: arm
// RUN: llvm-mc -arm-add-build-attributes -filetype=obj -triple=armv7a-linux-gnueabihf %s -o %t
// RUN: echo "SECTIONS { \
// RUN:       . = SIZEOF_HEADERS; \
// RUN:       .text 0x00011000 : { *(.text.*) } \
// RUN:       } \
// RUN:       sym = .;" > %t.script
// RUN: ld.lld --script %t.script %t -o %t2
// RUN: llvm-readobj --sections --symbols %t2 | FileCheck --check-prefix=CHECK-ELF %s
// RUN: llvm-objdump --no-show-raw-insn --start-address=0x11000 --stop-address=0x11048 -d %t2 | FileCheck %s

// An example of thunk generation that takes the maximum number of permitted
// passes to converge. We start with a set of branches of which all but one are
// in range. Any thunk added to extend the range of a branch is inserted in
// between the branches and the targets which knocks some more branches out
// of range. At the end of 9 passes of createThunks() every branch has a
// range extension thunk, allowing the final pass to check that no more thunks
// are required.
//
// As the size of the .text section changes 9 times, the symbol sym which
// depends on the size of .text will be updated 9 times. This test checks that
// any iteration limit to updating symbols does not limit thunk convergence.
// up to its pass limit without
//
// CHECK-ELF: Name: .text
// CHECK-ELF-NEXT:    Type: SHT_PROGBITS
// CHECK-ELF-NEXT:     Flags [
// CHECK-ELF-NEXT:       SHF_ALLOC
// CHECK-ELF-NEXT:       SHF_EXECINSTR
// CHECK-ELF-NEXT:     ]
// CHECK-ELF-NEXT:     Address: 0x11000
// CHECK-ELF-NEXT:     Offset: 0x1000
// CHECK-ELF-NEXT:     Size: 16777292
// CHECK-ELF:     Name: sym
// CHECK-ELF-NEXT:     Value: 0x101104C

// CHECK: 00011000 <_start>:
// CHECK-NEXT:    11000:       b.w     #14680132 <__Thumbv7ABSLongThunk_f2>
// CHECK-NEXT:    11004:       b.w     #14680128 <__Thumbv7ABSLongThunk_f2>
// CHECK-NEXT:    11008:       b.w     #14680128 <__Thumbv7ABSLongThunk_f3>
// CHECK-NEXT:    1100c:       b.w     #14680124 <__Thumbv7ABSLongThunk_f3>
// CHECK-NEXT:    11010:       b.w     #14680124 <__Thumbv7ABSLongThunk_f4>
// CHECK-NEXT:    11014:       b.w     #14680120 <__Thumbv7ABSLongThunk_f4>
// CHECK-NEXT:    11018:       b.w     #14680120 <__Thumbv7ABSLongThunk_f5>
// CHECK-NEXT:    1101c:       b.w     #14680116 <__Thumbv7ABSLongThunk_f5>
// CHECK-NEXT:    11020:       b.w     #14680116 <__Thumbv7ABSLongThunk_f6>
// CHECK-NEXT:    11024:       b.w     #14680112 <__Thumbv7ABSLongThunk_f6>
// CHECK-NEXT:    11028:       b.w     #14680112 <__Thumbv7ABSLongThunk_f7>
// CHECK-NEXT:    1102c:       b.w     #14680108 <__Thumbv7ABSLongThunk_f7>
// CHECK-NEXT:    11030:       b.w     #14680108 <__Thumbv7ABSLongThunk_f8>
// CHECK-NEXT:    11034:       b.w     #14680104 <__Thumbv7ABSLongThunk_f8>
// CHECK-NEXT:    11038:       b.w     #14680104 <__Thumbv7ABSLongThunk_f9>
// CHECK-NEXT:    1103c:       b.w     #14680100 <__Thumbv7ABSLongThunk_f9>
// CHECK-NEXT:    11040:       b.w     #14680100 <__Thumbv7ABSLongThunk_f10>
// CHECK-NEXT:    11044:       b.w     #14680096 <__Thumbv7ABSLongThunk_f10>


        .thumb
        .section .text.00, "ax", %progbits
        .globl _start
        .thumb_func
_start: b.w f2
        b.w f2
        b.w f3
        b.w f3
        b.w f4
        b.w f4
        b.w f5
        b.w f5
        b.w f6
        b.w f6
        b.w f7
        b.w f7
        b.w f8
        b.w f8
        b.w f9
        b.w f9
        b.w f10
        b.w f10

        .section .text.01, "ax", %progbits
        .space 14 * 1024 * 1024
// Thunks are inserted here, initially only 1 branch is out of range and needs
// a thunk. However the added thunk is 4-bytes in size which makes another
// branch out of range, which adds another thunk ...
        .section .text.02, "ax", %progbits
        .space (2 * 1024 * 1024) - 68
        .thumb_func
f2:     bx lr
        nop
        .thumb_func
f3:     bx lr
        nop
        .thumb_func
f4:     bx lr
        nop
        .thumb_func
f5:     bx lr
        nop
        .thumb_func
f6:     bx lr
        nop
        .thumb_func
f7:     bx lr
        nop
        .thumb_func
f8:     bx lr
        nop
        .thumb_func
f9:     bx lr
        nop
        .thumb_func
f10:     bx lr
        nop
