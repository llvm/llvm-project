# REQUIRES: aarch64

## Test that branch relocations with non-zero addends correctly target the
## actual function address, not the stub address. When a symbol is accessed
## via both a regular call (goes through stub) and a branch with addend
## (targeting an interior point), the addend must be applied to the real
## function VA, not the stub VA.
##
## This test uses -flat_namespace on a dylib, which makes locally-defined
## symbols interposable and thus accessible via stubs. This creates the
## scenario where a function is both defined locally AND in stubs.

# RUN: rm -rf %t; mkdir -p %t
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t/test.o
# RUN: %lld -arch arm64 -dylib -lSystem -flat_namespace %t/test.o -o %t/test.dylib

# RUN: llvm-objdump --no-print-imm-hex --macho -d %t/test.dylib | FileCheck %s

## With -flat_namespace, _target_func is interposable so regular calls go
## through stubs. But the branch with addend must go to the actual function
## address + addend, not stub + addend.
##
## Note: This means `bl _target_func` and `bl _target_func+16` could target
## different functions if interposition occurs at runtime. This is intentional:
## branching to an interior point implies reliance on the original function's
## layout, which an interposed replacement wouldn't preserve. There's no
## meaningful way to "interpose" an interior offset, so we target the original.

## _target_func layout:
##   offset 0:  nop
##   offset 4:  nop
##   offset 8:  nop
##   offset 12: nop
##   offset 16: mov w0, #42  <- this is what _target_func+16 should reach
##   offset 20: ret

## Verify _target_func layout and capture the address of the mov instruction
## (which is at _target_func + 16)
# CHECK-LABEL: _target_func:
# CHECK:       nop
# CHECK-NEXT:  nop
# CHECK-NEXT:  nop
# CHECK-NEXT:  nop
# CHECK-NEXT:  [[#%x,INTERIOR:]]:{{.*}}mov w0, #42
# CHECK-NEXT:  ret

## Verify the caller structure:
## - First bl goes to stub (marked with "symbol stub for:")
## - Second bl goes to [[INTERIOR]] (the _target_func+16 address captured above)
##
## The key assertion: the second bl MUST target _target_func+16 (INTERIOR),
## NOT stub+16. If the bug exists, it would target stub+16 which would be
## garbage (pointing past the stub section).
# CHECK-LABEL: _caller:
# CHECK:       bl {{.*}} symbol stub for: _target_func
# CHECK-NEXT:  bl 0x[[#INTERIOR]]
# CHECK-NEXT:  ret

.text
.globl _target_func, _caller
.p2align 2

_target_func:
  ## 4 nops = 16 bytes to offset 0x10
  nop
  nop
  nop
  nop
  ## This is at _target_func + 16
  mov w0, #42
  ret

_caller:
  ## Regular call to _target_func - goes through stub due to -flat_namespace
  bl _target_func
  ## Branch with addend - must go to actual function + 16, not stub + 16
  bl _target_func + 16
  ret

.subsections_via_symbols
