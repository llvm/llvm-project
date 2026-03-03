# REQUIRES: aarch64

## Test that mixing arm64 and arm64e object files is rejected.
## Even though both have CPU_TYPE_ARM64, arm64e requires pointer
## authentication and plain arm64 objects would cause PAC failures
## at runtime.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos  -o %t/arm64.o %t/lib.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/arm64e.o %t/lib.s

## Linking arm64e main with arm64 object should produce a warning about
## the architecture mismatch. The arm64 object is rejected, leading to
## an undefined symbol error.
# RUN: not %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem \
# RUN:   %t/test.o %t/arm64.o -o %t/out 2>&1 | FileCheck %s --check-prefix=WARN

# WARN: warning: {{.*}}arm64.o has architecture arm64 which is incompatible with target architecture arm64e (arm64e requires pointer authentication)
# WARN: error: undefined symbol: _helper

## Linking arm64e main with arm64e object should succeed silently.
# RUN: %no-arg-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -syslibroot %S/Inputs/MacOSX.sdk -lSystem -fatal_warnings \
# RUN:   %t/test.o %t/arm64e.o -o %t/out-ok

#--- test.s
.text
.globl _main

.p2align 2
_main:
  bl _helper
  ret

#--- lib.s
.text
.globl _helper

.p2align 2
_helper:
  ret
