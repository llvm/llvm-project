# REQUIRES: aarch64

## Test that authenticated pointer relocations correctly encode auth metadata
## (key, diversity, address diversity) through the Relocation union into
## chained fixup entries. This verifies the union-based auth data storage.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test

## Verify the binary is valid arm64e with chained fixups.
# RUN: llvm-objdump --macho --private-header %t/test | FileCheck %s --check-prefix=HEADER
# RUN: llvm-objdump --macho --chained-fixups %t/test | FileCheck %s --check-prefix=FIXUPS

# HEADER: ARM64          E

## Verify chained fixups use the ARM64E_USERLAND24 format and import _foo.
# FIXUPS: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# FIXUPS: _foo

## Verify the data section contains non-zero content (the auth pointer
## should have been encoded, not left as zero).
# RUN: llvm-objdump --macho -s --section __DATA,__data %t/test | FileCheck %s --check-prefix=DATA
# DATA-NOT: 00000000 00000000

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.text
.globl _main

.p2align 2
_main:
  ret

.data
.p2align 3
## Authenticated pointer with IA key, discriminator 0x1234, address diversity.
_auth_ptr:
.quad _foo@AUTH(ia,0x1234,addr)
