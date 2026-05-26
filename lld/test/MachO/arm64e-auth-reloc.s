# REQUIRES: aarch64

## Test ARM64_RELOC_AUTHENTICATED_POINTER handling.
## Verify that authenticated pointer relocations (@AUTH) are processed
## correctly and result in auth chained fixup entries.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test

## Verify the output is a valid arm64e binary (ARM64 with E subtype).
# RUN: llvm-objdump --macho --private-header %t/test | FileCheck %s --check-prefix=HEADER

# HEADER: ARM64          E

## Verify chained fixups contain the _foo import.
# RUN: llvm-objdump --macho --chained-fixups %t/test | FileCheck %s --check-prefix=FIXUPS

# FIXUPS: chained fixups header (LC_DYLD_CHAINED_FIXUPS)
# FIXUPS: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# FIXUPS: _foo

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
## Authenticated data pointer: sign _foo with IA key, discriminator 42,
## address diversity enabled.
.quad _foo@AUTH(ia,42,addr)
