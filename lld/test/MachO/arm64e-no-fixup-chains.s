# REQUIRES: aarch64

## Test that arm64e linking fails with a clear error when chained fixups
## are disabled via -no_fixup_chains, since dyld requires chained fixups
## for arm64e binaries.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: not %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   -no_fixup_chains %t/test.o -o %t/test 2>&1 | FileCheck %s

# CHECK: error: arm64e requires chained fixups; cannot use -no_fixup_chains

#--- test.s
.text
.globl _main

.p2align 2
_main:
  ret
