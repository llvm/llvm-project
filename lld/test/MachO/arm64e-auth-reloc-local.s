# REQUIRES: aarch64

## Verify ARM64_RELOC_AUTHENTICATED_POINTER targeting a local (non-extern)
## symbol round-trips its (key, diversity, addrDiv) metadata into the
## chained auth-rebase entry. Previously the non-extern path wrote the
## full 64-bit r.addend after auth decoding, silently zeroing the AuthInfo
## that shares those bits via union packing.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s

## Sanity-check the .o has a non-extern AUTH reloc (the assembler folds
## Lhidden into a section-relative reference to ltmp1).
# RUN: llvm-objdump -r %t/test.o | FileCheck %s --check-prefix=OBJ

# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib %t/test.o -o %t/test.dylib
# RUN: llvm-objdump --macho -s --section __DATA,__data %t/test.dylib | \
# RUN:   FileCheck %s --check-prefix=BYTES

# OBJ: ARM64_RELOC_AUTHENTICATED_POINTER ltmp{{[0-9]+}}

## The 8-byte auth-rebase entry is { target:32, diversity:16, addrDiv:1,
## key:2, next:11, bind:1, auth:1 }. With target = 0x4000 (Lhidden's
## runtime offset), diversity = 0x1234, key = DA (= 2), addrDiv = 1,
## bind = 0, auth = 1, next = 0, the encoded uint64 is 0x80051234_00004000.
## Printed by `objdump -s` (which loads each 4-byte word LE and formats
## %08x) that becomes "00004000 80051234". Without the union-preserving
## fix, the upper word would be "80000000" (auth flag set but metadata
## clobbered).
# BYTES: Contents of (__DATA,__data) section
# BYTES: {{0+}}8000 00004000 80051234

#--- test.s
.section __DATA,__const
.p2align 3
Lhidden:
  .quad 0xdeadbeef

.section __DATA,__data
.p2align 3
.globl _ptr
_ptr:
  .quad Lhidden@AUTH(da, 0x1234, addr)
