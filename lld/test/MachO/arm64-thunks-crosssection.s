# REQUIRES: aarch64

## This test verifies that thunks are created for branches between multiple
## code sections in the same segment when the combined span exceeds the
## branch range, even if each individual section is within range.
##
## The bug occurs when:
## (1) Section __text is within branch range (e.g., 64 MB < 128 MB)
## (2) Section __text_second is also within range (e.g., 64 MB < 128 MB)
## (3) Combined span exceeds branch range (128+ MB total)
## (4) Calls from early in __text to __text_second need thunks
##
## Without the fix, needsThunks() only considered individual section sizes,
## not the total span, causing BRANCH26 out of range errors.

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o
# RUN: %lld -arch arm64 -lSystem -o %t/out %t/input.o
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/out | FileCheck %s

# CHECK: Disassembly of section __TEXT,__text:

## _early_func is at the start of __text, it calls _far_func which is in
## __text_second section. This branch crosses sections and needs a thunk
## because the total span exceeds branch range.
# CHECK-LABEL: <_early_func>:
# CHECK:         bl {{.*}} <_far_func.thunk.0>
# CHECK:         bl {{.*}} <_helper>
# CHECK:         ret

# CHECK-LABEL: <_helper>:
# CHECK:         ret

## After padding, there's another function that also needs a thunk

## Verify thunk is created - it appears before _mid_func in output
# CHECK-LABEL: <_far_func.thunk.0>:
# CHECK:         adrp x16
# CHECK:         add  x16, x16
# CHECK:         br   x16

# CHECK-LABEL: <_mid_func>:
# CHECK:         bl {{.*}} <_far_func.thunk.0>
# CHECK:         ret

# CHECK: Disassembly of section __TEXT,__text_second:

# CHECK-LABEL: <_far_func>:
# CHECK:         ret


.text
.globl _main
.p2align 2
_main:
  bl _early_func
  ret

.globl _early_func
.p2align 2
_early_func:
  ## This call to _far_func crosses sections and exceeds branch range
  bl _far_func
  bl _helper
  ret

.globl _helper
.p2align 2
_helper:
  ret

## Pad __text section to ~64 MB
## 0x4000000 = 64 Mi = half the branch range
.space 0x4000000-0x20

.globl _mid_func
.p2align 2
_mid_func:
  bl _far_func
  ret

## More padding to push __text to ~128 MB
.space 0x4000000-0x10

## This is a second code section in __TEXT segment
.section __TEXT,__text_second,regular,pure_instructions

.globl _far_func
.p2align 2
_far_func:
  ret

## Add padding in second section to ensure total span exceeds branch range
.space 0x100000

.subsections_via_symbols
