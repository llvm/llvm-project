// REQUIRES: aarch64
// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: llvm-mc -filetype=obj -triple=aarch64 asm -o a.o
// RUN: ld.lld --shared --script=lds a.o -o out.so --defsym absolute=0xf0000000
// RUN: llvm-objdump -d --no-show-raw-insn out.so | FileCheck %s
// RUN: llvm-objdump -d --no-show-raw-insn out.so | FileCheck %s --check-prefix=CHECK-PADS
// RUN: llvm-mc -filetype=obj -triple=aarch64 shared -o shared.o
// RUN: ld.lld --shared -o shared.so shared.o
// RUN: ld.lld shared.so --script=lds a.o -o exe --defsym absolute=0xf0000000
// RUN: llvm-objdump -d --no-show-raw-insn exe | FileCheck %s --check-prefix=CHECK-EXE
// RUN: llvm-objdump -d --no-show-raw-insn exe | FileCheck %s --check-prefix=CHECK-PADS

/// Test thunk generation when destination does not have a BTI compatible
/// landing pad. Linker must generate landing pad sections for thunks that use
/// indirect branches.

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


/// Short thunks are direct branches so we don't need landing pads. Expect
/// all thunks to branch directly to target.
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
 bl .text.2 + 0x4 // fn2
 b  .text.2 + 0x4 // fn2
 bl fn1
 b  fn1
 bl fn3
 b  fn3
 bl fn4
 b  fn4
 bl via_plt
/// We cannot add landing pads for absolute symbols.
 bl absolute
/// padding so that we require thunks that can be placed after this section.
/// The thunks are close enough to the target to be short.
 .balign 8
 .space 0x1000

// CHECK-PADS-LABEL: <_start>:
// CHECK-PADS-NEXT: 10001000: bl      0x10002040
// CHECK-PADS-NEXT:           bl      0x10002044
// CHECK-PADS-NEXT:           bl      0x10002048
// CHECK-PADS-NEXT:           bl      0x1000204c
// CHECK-PADS-NEXT:           bl      0x10002050
// CHECK-PADS-NEXT:           bl      0x10002054
// CHECK-PADS-NEXT:           b       0x10002054
// CHECK-PADS-NEXT:           bl      0x10002058
// CHECK-PADS-NEXT:           b       0x10002058
// CHECK-PADS-NEXT:           bl      0x1000205c
// CHECK-PADS-NEXT:           b       0x1000205c
// CHECK-PADS-NEXT:           bl      0x10002060
// CHECK-PADS-NEXT:           b       0x10002060
// CHECK-PADS-NEXT:           bl      0x10002064
// CHECK-PADS-NEXT:           bl      0x10002068

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002040: b       0x18001000 <bti_c_target>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002044: b       0x18001008 <bti_j_target>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002048: b       0x18001010 <bti_jc_target>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 1000204c: b       0x18001018 <paciasp_target>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002050: b       0x18001020 <pacibsp_target>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002054: b       0x18001038 <fn2>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002058:       b       0x18001034 <fn1>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 1000205c:       b       0x18001040 <fn3>

// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 10002060:       b       0x18001050 <fn4>

// CHECK-LABEL: <__AArch64ADRPThunk_via_plt>:
// CHECK-NEXT: 10002064:       b       0x18001080 <via_plt@plt>

// CHECK-LABEL: <__AArch64ADRPThunk_absolute>:
// CHECK-NEXT: 10002068:       b       0x18001098 <absolute@plt>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002040: b       0x18001000 <bti_c_target>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002044: b       0x18001008 <bti_j_target>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002048: b       0x18001010 <bti_jc_target>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 1000204c: b       0x18001018 <paciasp_target>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002050: b       0x18001020 <pacibsp_target>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002054: b       0x18001038 <fn2>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002058: b       0x18001034 <fn1>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 1000205c: b       0x18001040 <fn3>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 10002060: b       0x18001050 <fn4>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_via_plt>:
// CHECK-EXE-NEXT: 10002064: b       0x18001080 <via_plt@plt>

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_absolute>:
// CHECK-EXE-NEXT: 10002068:   ldr     x16, 0x10002070 <__AArch64AbsLongThunk_absolute+0x8>
// CHECK-EXE-NEXT:             br      x16
// CHECK-EXE-NEXT: 00 00 00 f0 .word   0xf0000000
// CHECK-EXE-NEXT: 00 00 00 00 .word   0x00000000

.section .text.1, "ax", %progbits
/// These indirect branch targets already have a BTI compatible landing pad,
/// no alternative entry point required.
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

// CHECK-PADS-LABEL: <bti_c_target>:
// CHECK-PADS: 18001000:      bti     c
// CHECK-PADS-NEXT:           ret

// CHECK-PADS-LABEL: <bti_j_target>:
// CHECK-PADS-NEXT: 18001008: bti     j
// CHECK-PADS-NEXT:           ret

// CHECK-PADS-LABEL: <bti_jc_target>:
// CHECK-PADS-NEXT: 18001010: bti     jc
// CHECK-PADS-NEXT:           ret

// CHECK-PADS-LABEL: <paciasp_target>:
// CHECK-PADS-NEXT: 18001018: paciasp
// CHECK-PADS-NEXT:           ret

// CHECK-PADS-LABEL: <pacibsp_target>:
// CHECK-PADS-NEXT: 18001020: pacibsp
// CHECK-PADS-NEXT:           ret

/// These functions do not have BTI compatible landing pads. Expect linker
/// generated landing pads for indirect branch thunks.
.section .text.2, "ax", %progbits
.hidden fn1
.type fn1, %function
fn1:
 ret
.hidden fn2
.type fn2, %function
fn2:
 ret

// CHECK-PADS-LABEL: <__AArch64BTIThunk_>:
// CHECK-PADS-NEXT: 18001028: bti     c
// CHECK-PADS-NEXT:           b       0x18001038 <fn2>

// CHECK-PADS-LABEL: <__AArch64BTIThunk_>:
// CHECK-PADS-NEXT: 18001030: bti     c

// CHECK-PADS-LABEL: <fn1>:
// CHECK-PADS-NEXT: 18001034: ret

// CHECK-PADS-LABEL <fn2>:
// CHECK-PADS:      18001038: ret

/// Section with only one function at offset 0. Landing pad should be able to
/// fall through.
.section .text.3, "ax", %progbits
.hidden fn3
.type fn3, %function
fn3:
 ret

// CHECK-PADS-LABEL: <__AArch64BTIThunk_>:
// CHECK-PADS-NEXT: 1800103c: bti     c

// CHECK-PADS-LABEL: <fn3>:
// CHECK-PADS-NEXT: 18001040: ret

/// Section with only one function at offset 0, also with a high alignment
/// requirement. Check that we don't fall through into alignment padding.
.section .text.4, "ax", %progbits
.hidden fn4
.type fn4, %function
.balign 16
fn4:
 ret

// CHECK-PADS-LABEL: <__AArch64BTIThunk_>:
// CHECK-PADS:      18001044: bti     c
// CHECK-PADS-NEXT:           b       0x18001050 <fn4>
// CHECK-PADS-NEXT:           udf     #0x0

// CHECK-PADS-LABEL: <fn4>:
// CHECK-PADS-NEXT: 18001050: ret

.section .long_calls, "ax", %progbits
.global long_calls
.type long_calls, %function
long_calls:
/// Expect thunk to target as targets have BTI or implicit BTI.
 bl bti_c_target
 bl bti_j_target
 bl bti_jc_target
 bl paciasp_target
 bl pacibsp_target
/// Expect thunk to target a linker generated entry point with BTI landing pad.
/// Two calls to make sure only one landing pad is created.
 bl .text.2 + 0x4 // fn2
 b  .text.2 + 0x4 // fn2
/// fn2 before fn1 so that landing pad for fn1 can fall through.
 bl fn1
 b  fn1
 bl fn3
 b  fn3
 bl fn4
 b  fn4
/// PLT entries reachable via Thunks have a BTI c at the start of each entry
/// so no additional landing pad required.
 bl via_plt
/// We cannot add landing pads for absolute symbols.
 bl absolute
 .balign 8
/// PLT entries have BTI at start.
// CHECK-LABEL: <via_plt@plt>:
// CHECK-NEXT:           bti     c
// CHECK-NEXT:           adrp    x16, 0x30000000
// CHECK-NEXT:           ldr     x17, [x16, #0x1a0]
// CHECK-NEXT:           add     x16, x16, #0x1a0
// CHECK-NEXT:           br      x17
// CHECK-NEXT:           nop

// CHECK: <absolute@plt>:
// CHECK-NEXT:           bti     c
// CHECK-NEXT:           adrp    x16, 0x30000000
// CHECK-NEXT:           ldr     x17, [x16, #0x1a8]
// CHECK-NEXT:           add     x16, x16, #0x1a8
// CHECK-NEXT:           br      x17
// CHECK-NEXT:           nop

// CHECK-EXE-LABEL: <via_plt@plt>:
// CHECK-EXE-NEXT: 18001080: bti     c
// CHECK-EXE-NEXT:           adrp    x16, 0x30000000
// CHECK-EXE-NEXT:           ldr     x17, [x16, #0x1e8]
// CHECK-EXE-NEXT:           add     x16, x16, #0x1e8
// CHECK-EXE-NEXT:           br      x17
// CHECK-EXE-NEXT:           nop

// CHECK-LABEL: <long_calls>:
// CHECK-NEXT: 30000000: bl      0x30000040 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x3000004c <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x30000058 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x30000064 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x30000070 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x3000007c <__AArch64ADRPThunk_>
// CHECK-NEXT:           b       0x3000007c <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x30000088 <__AArch64ADRPThunk_>
// CHECK-NEXT:           b       0x30000088 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x30000094 <__AArch64ADRPThunk_>
// CHECK-NEXT:           b       0x30000094 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x300000a0 <__AArch64ADRPThunk_>
// CHECK-NEXT:           b       0x300000a0 <__AArch64ADRPThunk_>
// CHECK-NEXT:           bl      0x300000ac <__AArch64ADRPThunk_via_plt>
// CHECK-NEXT:           bl      0x300000b8 <__AArch64ADRPThunk_absolute>

/// bti_c_target.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT: 30000040: adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x0
// CHECK-NEXT:           br      x16
/// bti_j_target.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x8
// CHECK-NEXT:           br      x16
/// bti_jc_target.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x10
// CHECK-NEXT:           br      x16
/// paciasp_target.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x18
// CHECK-NEXT:           br      x16
/// pacibsp_target.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x20
// CHECK-NEXT:           br      x16
/// Landing pad for fn2.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x28
// CHECK-NEXT:           br      x16
/// Landing pad for fn1.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x30
// CHECK-NEXT:           br      x16
/// Landing pad for fn3.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x3c
// CHECK-NEXT:           br      x16
/// Landing pad for fn4.
// CHECK-LABEL: <__AArch64ADRPThunk_>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x44
// CHECK-NEXT:           br      x16

// CHECK-LABEL: <__AArch64ADRPThunk_via_plt>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x80
// CHECK-NEXT:           br      x16

// CHECK-LABEL: <__AArch64ADRPThunk_absolute>:
// CHECK-NEXT:           adrp    x16, 0x18001000 <bti_c_target>
// CHECK-NEXT:           add     x16, x16, #0x98
// CHECK-NEXT:           br      x16

// CHECK-EXE-LABEL: <long_calls>:
// CHECK-EXE-NEXT: 30000000: bl      0x30000040 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x30000050 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x30000060 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x30000070 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x30000080 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x30000090 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           b       0x30000090 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x300000a0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           b       0x300000a0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x300000b0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           b       0x300000b0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x300000c0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           b       0x300000c0 <__AArch64AbsLongThunk_>
// CHECK-EXE-NEXT:           bl      0x300000d0 <__AArch64AbsLongThunk_via_plt>
// CHECK-EXE-NEXT:           bl      0x300000e0 <__AArch64AbsLongThunk_absolute>

// CHECK-EXE-LABEL: 0000000030000040 <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000040: ldr     x16, 0x30000048 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     00 10 00 18   .word   0x18001000
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000050: ldr     x16, 0x30000058 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     08 10 00 18   .word   0x18001008
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000060: ldr     x16, 0x30000068 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     10 10 00 18   .word   0x18001010
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000070: ldr     x16, 0x30000078 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     18 10 00 18   .word   0x18001018
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000080: ldr     x16, 0x30000088 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     20 10 00 18   .word   0x18001020
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 30000090: ldr     x16, 0x30000098 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     28 10 00 18   .word   0x18001028
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 300000a0: ldr     x16, 0x300000a8 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     30 10 00 18   .word   0x18001030
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 300000b0: ldr     x16, 0x300000b8 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     3c 10 00 18   .word   0x1800103c
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_>:
// CHECK-EXE-NEXT: 300000c0: ldr     x16, 0x300000c8 <__AArch64AbsLongThunk_+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     44 10 00 18   .word   0x18001044
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_via_plt>:
// CHECK-EXE-NEXT: 300000d0: ldr     x16, 0x300000d8 <__AArch64AbsLongThunk_via_plt+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     80 10 00 18   .word   0x18001080
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

// CHECK-EXE-LABEL: <__AArch64AbsLongThunk_absolute>:
// CHECK-EXE-NEXT: 300000e0: ldr     x16, 0x300000e8 <__AArch64AbsLongThunk_absolute+0x8>
// CHECK-EXE-NEXT:           br      x16
// CHECK-EXE-NEXT:     00 00 00 f0   .word   0xf0000000
// CHECK-EXE-NEXT:     00 00 00 00   .word   0x00000000

//--- lds
PHDRS {
  low PT_LOAD FLAGS(0x1 | 0x4);
  mid PT_LOAD FLAGS(0x1 | 0x4);
  high PT_LOAD FLAGS(0x1 | 0x4);
}
SECTIONS {
  .rodata 0x10000000 : { *(.note.gnu.property) } :low
  .text_low : { *(.text.0) } :low
  .text 0x18001000 : { *(.text.*) } :mid
  .plt : { *(.plt) } :mid
  .text_high 0x30000000 : { *(.long_calls) } :high
}

//--- shared
.text
.global via_plt
.type via_plt, %function
via_plt:
 ret
