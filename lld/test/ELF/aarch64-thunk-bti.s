// REQUIRES: aarch64
// RUN: split-file %s %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 %t/asm -o %t.o
// RUN: ld.lld --shared --script=%t/lds %t.o -o %t.so --defsym absolute=0xf0000000
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x10000000 --stop-address=0x10000028 %t.so | FileCheck --check-prefix=CHECK-SO %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x17fc0028 --stop-address=0x17fc0050 %t.so | FileCheck --check-prefix=CHECK-SO2 %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x18000050 --stop-address=0x180000a8 %t.so | FileCheck --check-prefix=CHECK-BTI %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x180000b0 --stop-address=0x18000100 %t.so | FileCheck --check-prefix=CHECK-SO3 %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x30000000 --stop-address=0x300000c0 %t.so | FileCheck --check-prefix=CHECK-SO4 %s
// RUN: rm %t.so
// RUN: llvm-mc -filetype=obj -triple=aarch64 %t/shared -o %tshared.o
// RUN: ld.lld --shared -o %tshared.so %tshared.o
// RUN: ld.lld %tshared.so --script=%t/lds %t.o -o %t.exe --defsym absolute=0xf0000000
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x10000000 --stop-address=0x10000028 %t.exe | FileCheck --check-prefix=CHECK-EXE %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x17fc0028 --stop-address=0x17fc0050 %t.exe | FileCheck --check-prefix=CHECK-EXE2 %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x18000050 --stop-address=0x180000a8 %t.exe | FileCheck --check-prefix=CHECK-BTI %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x180000b0 --stop-address=0x180000e8 %t.exe | FileCheck --check-prefix=CHECK-EXE3 %s
// RUN: llvm-objdump -d --no-show-raw-insn --print-imm-hex --start-address=0x30000000 --stop-address=0x300000ec %t.exe | FileCheck --check-prefix=CHECK-EXE4 %s
// RUN: rm %t.o %tshared.o %tshared.so

/// Test thunk generation when destination does not have
/// a BTI compatible landing pad. Linker must generate
/// landing pad sections for thunks that use indirect
/// branches.

//--- asm
.section ".note.gnu.property", "a"
.p2align 3
.long 4
.long 0x10
.long 0x5
.asciz "GNU"

/// Enable BTI.
.long 0xc0000000 // GNU_PROPERTY_AARCH64_FEATURE_1_AND.
.long 4
.long 1          // GNU_PROPERTY_AARCH64_FEATURE_1_BTI.
.long 0


/// Short thunks are direct branches so we don't
/// need landing pads. Expect all thunks to branch
/// to target.
.section .text.0, "ax", %progbits
.balign 0x1000
.global _start
.type _start, %function
_start:
 bl bti_c_target
 bl bti_j_target
 bl bti_jc_target
 bl paciasp_target
 bl pacibsp_target
 bl fn1
 bl fn2
 bl fn3
 bl  fn4
 bl via_plt

// CHECK-SO-LABEL: <_start>:
// CHECK-SO-NEXT: 10000000: bl      0x17fc0028 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc002c <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0030 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0034 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0038 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc003c <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0040 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0044 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc0048 <__AArch64ADRPThunk_>
// CHECK-SO-NEXT:           bl      0x17fc004c <__AArch64ADRPThunk_via_plt>

// CHECK-SO2:      <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:17fc0028: b       0x18000050 <bti_c_target>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000058 <bti_j_target>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000060 <bti_jc_target>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000068 <paciasp_target>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000070 <pacibsp_target>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000088 <fn1>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x1800008c <fn2>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x18000094 <fn3>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO2-NEXT:          b       0x180000a0 <fn4>
// CHECK-SO2-EMPTY:
// CHECK-SO2-NEXT: <__AArch64ADRPThunk_via_plt>:
// CHECK-SO2-NEXT:          b       0x180000d0 <fn4+0x30>

// CHECK-EXE-LABEL: <_start>:
// CHECK-EXE-NEXT: 10000000: bl      0x17fc0028 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc002c <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0030 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0034 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0038 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc003c <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0040 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0044 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc0048 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x17fc004c <__AArch64AbsLongThunk_via_plt>

// CHECK-EXE2: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT: 17fc0028: b       0x18000050 <bti_c_target>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000058 <bti_j_target>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000060 <bti_jc_target>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000068 <paciasp_target>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000070 <pacibsp_target>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000088 <fn1>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x1800008c <fn2>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x18000094 <fn3>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE2-NEXT:           b       0x180000a0 <fn4>
// CHECK-EXE2-EMPTY:
// CHECK-EXE2-NEXT: <__AArch64AbsLongThunk_via_plt>:
// CHECK-EXE2-NEXT:           b       0x180000d0 <fn4+0x30>

// padding to put .text.short out of
// range, but with enough gaps to
// place a pool that can short thunk
// to targets.
.section .text.1, "ax", %progbits
.space 0x2000000

.section .text.2, "ax", %progbits
.space 0x2000000

.section .text.3, "ax", %progbits
.space 0x2000000

.section .text.4, "ax", %progbits
.space 0x2000000 - 0x40000

.section .text.5, "ax", %progbits
.space 0x40000

.section .text.6, "ax", %progbits
/// Indirect branch targets have BTI compatible landing pad
/// No alternative entry point required.
.hidden bti_c_target
.type bti_c_target, %function
bti_c_target:
 bti c
 ret

.hidden bti_j_target
.type bti_j_target, %function
bti_j_target:
 bti j
 ret

.hidden bti_jc_target
.type bti_jc_target, %function
bti_jc_target:
 bti jc
 ret

.hidden paciasp_target
.type paciasp_target, %function
paciasp_target:
 paciasp
 ret

.hidden pacibsp_target
.type pacibsp_target, %function
pacibsp_target:
 pacibsp
 ret

// CHECK-BTI-LABEL: <bti_c_target>:
// CHECK-BTI-NEXT: 18000050: bti     c
// CHECK-BTI-NEXT:           ret
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <bti_j_target>:
// CHECK-BTI-NEXT:           bti     j
// CHECK-BTI-NEXT:           ret
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <bti_jc_target>:
// CHECK-BTI-NEXT:           bti     jc
// CHECK-BTI-NEXT:           ret
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <paciasp_target>:
// CHECK-BTI-NEXT:           paciasp
// CHECK-BTI-NEXT:           ret
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <pacibsp_target>:
// CHECK-BTI-NEXT:           pacibsp
// CHECK-BTI-NEXT:           ret

/// functions do not have BTI compatible landing pads.
/// Expect linker generated landing pads.
.section .text.7, "ax", %progbits
.hidden fn1
.type fn1, %function
fn1:
 ret
.hidden fn2
.type fn2, %function
fn2:
 ret

// CHECK-BTI: <__AArch64BTIThunk_>:
// CHECK-BTI-NEXT: 18000078: bti     c
// CHECK-BTI-NEXT:           b       0x18000088 <fn1>
// CHECK-BTI-EMPTY:
// CHECK-BTI-NEXT: <__AArch64BTIThunk_>:
// CHECK-BTI-NEXT:           bti     c
// CHECK-BTI-NEXT:           b       0x1800008c <fn2>
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <fn1>:
// CHECK-BTI-NEXT:           ret
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL:  <fn2>:
// CHECK-BTI-NEXT:           ret

/// Section with only one function at offset 0. Landing pad should
/// be able to fall through.
.section .text.8, "ax", %progbits
.hidden fn3
.type fn3, %function
fn3:
 ret

// CHECK-BTI: <__AArch64BTIThunk_>:
// CHECK-BTI-NEXT: 18000090: bti     c
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <fn3>:
// CHECK-BTI-NEXT:           ret

/// Section with only one function at offset 0, also with a high
/// alignment requirement. Check that we don't fall through into
/// alignment padding.
.section .text.9, "ax", %progbits
.balign 16
.hidden fn4
.type fn4, %function
fn4:
 ret

// CHECK-BTI: <__AArch64BTIThunk_>:
// CHECK-BTI-NEXT: 18000098: bti     c
// CHECK-BTI-NEXT:           b       0x180000a0 <fn4>
// CHECK-BTI-EMPTY:
// CHECK-BTI-LABEL: <fn4>:
// CHECK-BTI-NEXT:           ret

/// PLT has bti c.
// CHECK-SO3: 180000b0: bti     c
// CHECK-SO3-NEXT: stp     x16, x30, [sp, #-0x10]!
// CHECK-SO3-NEXT: adrp    x16, 0x30000000 <via_plt+0x30000000>
// CHECK-SO3-NEXT: ldr     x17, [x16, #0x2d0]
// CHECK-SO3-NEXT: add     x16, x16, #0x2d0
// CHECK-SO3-NEXT: br      x17
// CHECK-SO3-NEXT: nop
// CHECK-SO3-NEXT: nop
// CHECK-SO3-NEXT: bti     c
// CHECK-SO3-NEXT: adrp    x16, 0x30000000 <via_plt+0x30000000>
// CHECK-SO3-NEXT: ldr     x17, [x16, #0x2d8]
// CHECK-SO3-NEXT: add     x16, x16, #0x2d8
// CHECK-SO3-NEXT: br      x17
// CHECK-SO3-NEXT: nop
// CHECK-SO3-NEXT: bti     c
// CHECK-SO3-NEXT: adrp    x16, 0x30000000 <via_plt+0x30000000>
// CHECK-SO3-NEXT: ldr     x17, [x16, #0x2e0]
// CHECK-SO3-NEXT: add     x16, x16, #0x2e0
// CHECK-SO3-NEXT: br      x17
// CHECK-SO3-NEXT: nop

// CHECK-EXE3: 180000b0: bti     c
// CHECK-EXE3-NEXT:      stp     x16, x30, [sp, #-0x10]!
// CHECK-EXE3-NEXT:      adrp    x16, 0x30000000 <via_plt+0x30000000>
// CHECK-EXE3-NEXT:      ldr     x17, [x16, #0x320]
// CHECK-EXE3-NEXT:      add     x16, x16, #0x320
// CHECK-EXE3-NEXT:      br      x17
// CHECK-EXE3-NEXT:      nop
// CHECK-EXE3-NEXT:      nop
// CHECK-EXE3-NEXT:      bti     c
// CHECK-EXE3-NEXT:      adrp    x16, 0x30000000 <via_plt+0x30000000>
// CHECK-EXE3-NEXT:      ldr     x17, [x16, #0x328]
// CHECK-EXE3-NEXT:      add     x16, x16, #0x328
// CHECK-EXE3-NEXT:      br      x17
// CHECK-EXE3-NEXT:      nop

.section .long_calls, "ax", %progbits
.global long_calls
.type long_calls, %function
/// Expect thunk to target as targets have
/// BTI or implicit BTI.
 bl bti_c_target
 bl bti_j_target
 bl bti_jc_target
 bl paciasp_target
 bl pacibsp_target
/// Expect thunk to target a linker generated
/// entry point with BTI landing pad.
/// Two calls to make sure only one landing
/// pad is created.
 bl fn1
 b  fn1
 bl fn2
 b  fn2
 bl fn3
 b  fn3
 bl fn4
 b  fn4
/// PLT entries reachable via Thunks have a
/// BTI c at the start of each entry so no
/// additional landing pad required.
 bl via_plt
/// We cannot add landing pads for absolute
/// symbols.
 bl absolute

// CHECK-SO4-LABEL: <.text_high>:
// CHECK-SO4-NEXT: 30000000: bl      0x3000003c <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000048 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000054 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000060 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x3000006c <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000078 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           b       0x30000078 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000084 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           b       0x30000084 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x30000090 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           b       0x30000090 <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x3000009c <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           b       0x3000009c <__AArch64ADRPThunk_>
// CHECK-SO4-NEXT:           bl      0x300000a8 <__AArch64ADRPThunk_via_plt>
// CHECK-SO4-NEXT:           bl      0x300000b4 <__AArch64ADRPThunk_absolute>
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x50
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x58
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x60
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x68
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x70
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x78
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x80
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x90
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0x98
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_via_plt>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0xd0
// CHECK-SO4-NEXT:           br      x16
// CHECK-SO4-EMPTY:
// CHECK-SO4-NEXT: <__AArch64ADRPThunk_absolute>:
// CHECK-SO4-NEXT:           adrp    x16, 0x18000000 <__AArch64ADRPThunk_via_plt+0x3ffb4>
// CHECK-SO4-NEXT:           add     x16, x16, #0xe8
// CHECK-SO4-NEXT:           br      x16

// CHECK-EXE4-LABEL: <.text_high>:
// CHECK-EXE4-NEXT: 30000000: bl      0x3000003c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000004c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000005c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000006c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000007c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000008c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           b       0x3000008c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x3000009c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           b       0x3000009c <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x300000ac <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           b       0x300000ac <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x300000bc <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           b       0x300000bc <__AArch64AbsLongThunk_>
// CHECK-EXE4-NEXT:           bl      0x300000cc <__AArch64AbsLongThunk_via_plt>
// CHECK-EXE4-NEXT:           bl      0x300000dc <__AArch64AbsLongThunk_absolute>
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000044 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           50 00 00 18       .word   0x18000050
// CHECK-EXE4-NEXT:           00 00 00 00       .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000054 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           58 00 00 18       .word   0x18000058
// CHECK-EXE4-NEXT:           00 00 00 00       .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT:  <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000064 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           60 00 00 18       .word   0x18000060
// CHECK-EXE4-NEXT:           00 00 00 00       .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT:  <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000074 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           68 00 00 18       .word   0x18000068
// CHECK-EXE4-NEXT:           00 00 00 00       .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000084 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           70 00 00 18        .word   0x18000070
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x30000094 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           78 00 00 18        .word   0x18000078
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x300000a4 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           80 00 00 18        .word   0x18000080
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x300000b4 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           90 00 00 18        .word   0x18000090
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x300000c4 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           98 00 00 18        .word   0x18000098
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_via_plt>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x300000d4 <__AArch64AbsLongThunk_via_plt+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           d0 00 00 18        .word   0x180000d0
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000
// CHECK-EXE4-EMPTY:
// CHECK-EXE4-NEXT: <__AArch64AbsLongThunk_absolute>:
// CHECK-EXE4-NEXT:           ldr     x16, 0x300000e4 <__AArch64AbsLongThunk_absolute+0x8>
// CHECK-EXE4-NEXT:           br      x16
// CHECK-EXE4-NEXT:           00 00 00 f0        .word   0xf0000000
// CHECK-EXE4-NEXT:           00 00 00 00        .word   0x00000000

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
    .text_low 0x10000000 : {
        *(.text.*)
	*(.plt)
    } :low
    .text_high 0x30000000 : { *(.long_calls) } :high
}

//--- shared
.text
.global via_plt
.type via_plt, %function
via_plt:
 ret
