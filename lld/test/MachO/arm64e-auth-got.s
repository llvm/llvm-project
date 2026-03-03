# REQUIRES: aarch64

## A symbol that is both branched to (BRANCH stub target) and address-taken
## (GOT_LOAD) on arm64e must land in BOTH __auth_got and __got. The auth
## slot feeds the stub (signed pointer); the regular slot feeds the
## paciza-based address-of operation.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib -install_name @executable_path/libfoo.dylib %t/foo.o \
# RUN:   -o %t/libfoo.dylib
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test

# RUN: llvm-objdump --macho --section-headers %t/test \
# RUN:   | FileCheck %s --check-prefix=SECT
# RUN: llvm-objdump --macho --chained-fixups %t/test \
# RUN:   | FileCheck %s --check-prefix=CHAIN
# RUN: llvm-objdump --macho -s --section __DATA_CONST,__auth_got \
# RUN:   --section __DATA_CONST,__got %t/test \
# RUN:   | FileCheck %s --check-prefix=BYTES

## __auth_got is laid out before __got (Writer places the signed slots first
## so the chain's `next` pointer steps from the auth slot to the regular slot).
# SECT:      __auth_got    00000008 [[#%x,AUTH:]] DATA
# SECT-NEXT: __got         00000008 [[#%x,GOT:]] DATA

## Exactly one import — the same _foo serves both slots.
# CHAIN:      imports_count  = 1
# CHAIN:      pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# CHAIN:      dyld chained import[0]
# CHAIN:        name_offset = 0 (_foo)

## __auth_got entry encodes an auth-bind24 (auth=1, bind=1, key=IA, addrDiv=1)
## with next=1 so the chain advances to the __got slot 8 bytes later.
## __got entry is a plain bind24 (auth=0, bind=1) terminating the chain.
## Top byte 0xc0 in the auth slot = bind|auth bits set; 0x40 in the plain
## slot = bind only.
# BYTES-LABEL: Contents of (__DATA_CONST,__auth_got) section
# BYTES:       {{0+}}4000 00000000 c0090000
# BYTES-LABEL: Contents of (__DATA_CONST,__got) section
# BYTES:       {{0+}}4008 00000000 40000000

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.text
.globl _main

.p2align 2
_main:
  ## Call _foo — emits a stub that loads its target from __auth_got.
  bl _foo

  ## Take the address of _foo — GOT_LOAD lowers to a plain __got entry.
  adrp x0, _foo@GOTPAGE
  ldr  x0, [x0, _foo@GOTPAGEOFF]

  ret
