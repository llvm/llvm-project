# REQUIRES: aarch64

## Test arm64e chained fixup pointer format selection.
## macOS 12.0+ should use DYLD_CHAINED_PTR_ARM64E_USERLAND24;
## older deployment targets use DYLD_CHAINED_PTR_ARM64E when
## chained fixups are explicitly requested.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/foo.o %t/foo.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib -install_name @executable_path/libfoo.dylib %t/foo.o -o %t/libfoo.dylib

## Link with macOS 13.0 (>= 12.0) — should use USERLAND24.
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test-new
# RUN: llvm-objdump --macho --chained-fixups %t/test-new | \
# RUN:   FileCheck %s --check-prefix=USERLAND24

## Link with macOS 11.0 (< 12.0) with explicit -fixup_chains
## — should use plain DYLD_CHAINED_PTR_ARM64E format.
# RUN: %no-arg-lld -arch arm64e -platform_version macos 11.0 11.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -fixup_chains \
# RUN:   %t/libfoo.dylib %t/test.o -o %t/test-old
# RUN: llvm-objdump --macho --chained-fixups %t/test-old | \
# RUN:   FileCheck %s --check-prefix=PLAIN

# USERLAND24: chained fixups header (LC_DYLD_CHAINED_FIXUPS)
# USERLAND24: pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)

# PLAIN: chained fixups header (LC_DYLD_CHAINED_FIXUPS)
# PLAIN: pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)

#--- foo.s
.globl _foo
_foo:
  ret

#--- test.s
.text
.globl _main

.p2align 2
_main:
  bl _foo
  ret

.data
.p2align 3
.quad _foo
