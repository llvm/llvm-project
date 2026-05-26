# REQUIRES: aarch64

## Smoke test: the high8 channel of an arm64e chained rebase round-trips a
## non-zero byte from the addend in both USERLAND24 and legacy ARM64E formats.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s

## USERLAND24 (macOS 13.0 — implicit chained fixups).
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -dylib %t/test.o -o %t/test-u24.dylib
# RUN: llvm-objdump --macho --chained-fixups %t/test-u24.dylib | \
# RUN:   FileCheck %s --check-prefix=U24-FMT
# RUN: llvm-objdump --macho -s --section __DATA,__data %t/test-u24.dylib | \
# RUN:   FileCheck %s --check-prefix=U24-BYTES

## Legacy ARM64E (macOS 11.0 + explicit -fixup_chains).
# RUN: %no-arg-lld -arch arm64e -platform_version macos 11.0 11.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -fixup_chains \
# RUN:   -dylib %t/test.o -o %t/test-legacy.dylib
# RUN: llvm-objdump --macho --chained-fixups %t/test-legacy.dylib | \
# RUN:   FileCheck %s --check-prefix=LEGACY-FMT
# RUN: llvm-objdump --macho -s --section __DATA,__data %t/test-legacy.dylib | \
# RUN:   FileCheck %s --check-prefix=LEGACY-BYTES

# U24-FMT:    pointer_format = 12 (DYLD_CHAINED_PTR_ARM64E_USERLAND24)
# LEGACY-FMT: pointer_format = 1 (DYLD_CHAINED_PTR_ARM64E)

## Encoded uint64 is 0x0007F800_00004008 (target=0x4008, high8=0xFF, all other
## fields 0). objdump -s prints each 4-byte word in LE host order, so the bytes
## appear as "00004008 0007f800".
# U24-BYTES:    Contents of (__DATA,__data) section
# U24-BYTES:    {{0+}}4000 00004008 0007f800
# LEGACY-BYTES: Contents of (__DATA,__data) section
# LEGACY-BYTES: {{0+}}4000 00004008 0007f800

#--- test.s
.section __DATA,__data
.p2align 3
.globl _ptr
_ptr:
  .quad _target + 0xff00000000000000

.p2align 3
_target:
  .quad 0
