# REQUIRES: aarch64

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -lSystem -arch arm64 -o %t %t.o
# RUN: llvm-objdump --macho -d --full-contents --section-headers %t | FileCheck %s

## FIXME: Even though we have reserved a GOT slot for _foo due to
## POINTER_TO_GOT, we should still be able to relax this GOT_LOAD reference to
## it.
# CHECK:      _main:
# CHECK-NEXT: adrp x8, [[#]] ; 0x100004000
# CHECK-NEXT: ldr  x8, [x8]
# CHECK-NEXT: ret

# CHECK: Idx   Name          Size     VMA              Type
# CHECK: [[#]] __got         00000008 0000000100004000 DATA
# CHECK: [[#]] __data        00000004 0000000100008000 DATA

## The relocated data should contain the difference between the addresses of
## __data and __got in little-endian form.
# CHECK:       Contents of section __DATA,__data:
# CHECK-NEXT:  100008000 00c0ffff

.globl _main, _foo
.p2align 2
_main:
  adrp x8, _foo@GOTPAGE
  ldr  x8, [x8, _foo@GOTPAGEOFF]
  ret

.p2align 2
_foo:
  ret

.data
.long _foo@GOT - .
