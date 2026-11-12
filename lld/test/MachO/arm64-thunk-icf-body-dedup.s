# REQUIRES: aarch64

# RUN: rm -rf %t; mkdir %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/input.o

## Verify that ICF body-folded symbols share a single branch-extension thunk
## rather than each getting its own.
# RUN: %lld -arch arm64 -lSystem -o %t/icf-all %t/input.o --icf=all -map %t/icf-all.map
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/icf-all | FileCheck %s --check-prefix=BODY
# RUN: FileCheck %s --input-file %t/icf-all.map --check-prefix=BODY-MAP

## Both calls in _main branch to the same thunk address.
# BODY-LABEL: <_main>:
# BODY:       bl 0x[[#%x, THUNK:]] <_target_a.thunk.0>
# BODY-NEXT:  bl 0x[[#%x, THUNK]] <_target_a.thunk.0>

## Only one thunk should be created for the folded functions.
# BODY-MAP:     .thunk.0
# BODY-MAP-NOT: .thunk.0

## Verify that safe_thunks ICF still creates separate branch-extension thunks
## when needed.
# RUN: %lld -arch arm64 -lSystem -o %t/icf-safe %t/input.o --icf=safe_thunks
# RUN: llvm-objdump --no-print-imm-hex -d --no-show-raw-insn %t/icf-safe | FileCheck %s --check-prefix=SAFE

## Each call gets its own branch-extension thunk.
# SAFE-LABEL: <_main>:
# SAFE:       bl 0x[[#%x, THUNK_A:]] <_target_a.thunk.0>
# SAFE-NEXT:  bl 0x[[#%x, THUNK_B:]] <_target_b.thunk.0>

.subsections_via_symbols

.addrsig
.addrsig_sym _target_a
.addrsig_sym _target_b

.text

## A unique function placed before _target_a so that the assembler's automatic
## ltmp0 symbol lands here to make the test more readable.
.globl _unique_func
.p2align 2
_unique_func:
  mov w0, #1
  ret

.globl _target_a
.p2align 2
_target_a:
  mov w0, #42
  ret

.globl _target_b
.p2align 2
_target_b:
  mov w0, #42
  ret

.globl _spacer
.p2align 2
_spacer:
  .space 0x8000000
  ret

.globl _main
.p2align 2
_main:
  bl _target_a
  bl _target_b
  ret

## With safe_thunks, _target_b's ICF thunk is a synthetic section appended
## after all regular inputs. This spacer pushes it out of branch range from
## _main so it also needs a branch-extension thunk.
.globl _spacer2
.p2align 2
_spacer2:
  .space 0x8000000
  mov w0, #0
  ret
