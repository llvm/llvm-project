# REQUIRES: aarch64

## Test that ICF works correctly on arm64e binaries containing
## authenticated pointer relocations. Identical functions should
## still be folded, and auth relocations in data sections should
## not cause ICF to crash or misbehave.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   --icf=all %t/test.o -o %t/test
# RUN: llvm-objdump --macho --syms %t/test | FileCheck %s

## _func_a and _func_b have identical bodies (just ret) and should be
## folded by ICF even in an arm64e binary with auth data present.
# CHECK-DAG: [[#%x,FUNC:]] l     F __TEXT,__text _func_a
# CHECK-DAG: [[#%x,FUNC]]  l     F __TEXT,__text _func_b

#--- test.s
.subsections_via_symbols

.text
.globl _main
.p2align 2
_main:
  ret

.globl _target
.p2align 2
_target:
  ret

## Two identical functions — should be folded.
.p2align 2
_func_a:
  ret

.p2align 2
_func_b:
  ret

## Auth data in a data section — ensures auth relocs don't
## interfere with ICF processing.
.data
.p2align 3
_auth_ptr:
  .quad _target@AUTH(ia,42,addr)
