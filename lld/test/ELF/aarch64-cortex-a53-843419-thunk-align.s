// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64 %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:                  .text 0x10000 : { \
// RUN:                  *(.text.01) ; \
// RUN:                  . += 0x8000000 ;  \
// RUN:                  *(.text.02) } \
// RUN:                  .foo : { *(.foo_sec) } } " > %t.script
// RUN: ld.lld -pie --fix-cortex-a53-843419 --script=%t.script %t.o -o %t2
// RUN: llvm-objdump --no-show-raw-insn --triple=aarch64-linux-gnu -d %t2


/// %t2 is > 128 Megabytes, so delete it early.
// RUN: rm %t2

/// Test case that for an OutputSection larger than the ThunkSectionSpacing
/// --fix-cortex-a53-843419 will cause the size of the ThunkSection to be
/// rounded up to the nearest 4KiB

 .section .text.01, "ax", %progbits
 .balign 4096
 .globl _start
 .type _start, %function
_start:
/// Range extension thunk needed, due to linker script
 bl far_away
 .space 4096 - 12

/// Erratum sequence
 .globl t3_ff8_ldr
 .type t3_ff8_ldr, %function
t3_ff8_ldr:
 adrp x0, dat
 ldr x1, [x1, #0]
 ldr x0, [x0, :lo12:dat]
 ret

/// Expect thunk and patch to be inserted here
// CHECK: 0000000000011008 __AArch64ADRPThunk_far_away:
// CHECK-NEXT: 11008: adrp    x16, #134221824
// CHECK-NEXT:        add     x16, x16, #16
// CHECK-NEXT:        br      x16
// CHECK: 0000000000012008 __CortexA53843419_11000:
// CHECK-NEXT: 12008: ldr     x0, [x0, #168]
// CHECK-NEXT:        b       #-4104 <t3_ff8_ldr+0xc>

 .section .text.02, "ax", %progbits
 .globl far_away
 .type far_away, function
far_away:
 bl _start
 ret
/// Expect thunk for _start not to have size rounded up to 4KiB as it is at
/// the end of the OutputSection
// CHECK: 0000000008012010 far_away:
// CHECK-NEXT:  8012010: bl      #8
// CHECK-NEXT:           ret
// CHECK: 0000000008012018 __AArch64ADRPThunk__start:
// CHECK-NEXT:  8012018: adrp    x16, #-134225920
// CHECK-NEXT:           add     x16, x16, #0
// CHECK-NEXT:           br      x16
// CHECK: 0000000008012024 foo:
// CHECK-NEXT:  8012024: ret
 .section .foo_sec, "ax", %progbits
 .globl foo
 .type foo, function
foo:
  ret


 .section .data
 .balign 8
 .globl dat
dat:    .quad 0
