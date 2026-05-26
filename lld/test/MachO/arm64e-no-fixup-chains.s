# REQUIRES: aarch64

## Test that arm64e linking with -no_fixup_chains produces a warning
## and uses chained fixups anyway, since dyld requires them for arm64e.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -no_fixup_chains %t/test.o -o %t/test 2>&1 | FileCheck %s

# CHECK: warning: -no_fixup_chains is incompatible with arm64e; using chained fixups

## Verify the output still has chained fixups.
# RUN: llvm-objdump --macho --all-headers %t/test | FileCheck %s --check-prefix=HEADERS

# HEADERS: LC_DYLD_CHAINED_FIXUPS

#--- test.s
.text
.globl _main

.p2align 2
_main:
  ret
