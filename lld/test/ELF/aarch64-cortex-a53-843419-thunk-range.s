// REQUIRES: aarch64
// RUN: llvm-mc -filetype=obj -triple=aarch64-none-linux --asm-macro-max-nesting-depth=40000 %s -o %t.o
// RUN: echo "SECTIONS { \
// RUN:          .text1 0x10000 : { *(.text.*) } \
// RUN:          .text2 0xf010000 : { *(.target) } } " > %t.script
// RUN: ld.lld --script %t.script -fix-cortex-a53-843419 %t.o -o %t2 --print-map 2>&1 | FileCheck %s

/// We use %(\parameter) to evaluate expression, which requires .altmacro.
 .altmacro

/// Test to reproduce the conditions that trigger R_AARCH64_JUMP26 out of range
/// errors in pr44071. We create a large number of patches and thunks, with an
/// LLD with the fault, the patches will be inserted after the thunks and due
/// to the size of the thunk section some of the patches go out of range.
/// With a fixed LLD the patches are inserted before the thunks.

// CHECK: <internal>:(.text.patch)
// CHECK: <internal>:(.text.thunk)

/// Macro to generate the cortex-a53-843419 patch sequence
 .macro ERRATA from, to
   .balign 4096
   .space 4096 - 8
   adrp x0, dat1
   ldr x1, [x1, #0]
   ldr x0, [x0, :got_lo12:dat1]
   ret
   .if (\to-\from)
     ERRATA %(\from+1),\to
   .endif
 .endm

 .section .text.01, "ax", %progbits
 .balign 4096
 .globl _start
 .type _start, %function
 .space 4096 - 8
_start:
/// Generate lots of patches.
 ERRATA 0, 4000

 .macro CALLS from, to
   bl far\from
   .if (\to-\from)
     CALLS %(\from+1),\to
   .endif
 .endm

 /// Generate long range thunks. These are inserted before the patches. Generate
 /// a sufficient number such that the patches must be placed before the
 /// .text.thunk section, and if they aren't some of the patches go out of
 /// range.
 .section .text.02, "ax", %progbits
 .global func
 .type func, %function
func:
 CALLS 0, 20000

 .section .text.03, "ax", %progbits
 .global space1
space1:
 .space (1024 * 1024 * 96) + (120 * 4 * 1024)
 .balign 4096

 .section .text.04, "ax", %progbits
 .global space2
space2:
 .space 1024 * 1024

 .macro DEFS from, to
   .global far\from
   .type far\from, %function
far\from:
   ret
   .if (\to-\from)
     DEFS %(\from+1),\to
   .endif
 .endm

 /// Define the thunk targets
 .section .target, "ax", %progbits
 DEFS 0, 20000

 .data
 .global dat1
dat1:
 .xword 0
