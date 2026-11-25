# REQUIRES: aarch64
## arm64 and arm64e share a CPU type but are ABI incompatible: an arm64e link
## ignores arm64 objects and an arm64 link ignores arm64e objects, and both
## reject them under -arch_errors_fatal.

# RUN: rm -rf %t; split-file %s %t
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/test.o %t/test.s
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/arm64.o %t/lib.s
# RUN: llvm-mc -filetype=obj -triple=arm64e-apple-macos -o %t/arm64e.o %t/lib.s

# RUN: not %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   %t/test.o %t/arm64.o -o /dev/null 2>&1 | FileCheck %s --check-prefix=WARN
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/test.o %t/arm64.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 %t/test.o \
# RUN:   %t/arm64e.o -o /dev/null

# RUN: llvm-mc -filetype=obj -triple=arm64-apple-macos -o %t/test-arm64.o \
# RUN:   %t/test.s
# RUN: not %no-fatal-warnings-lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   %t/test-arm64.o %t/arm64e.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARN-ARM64
# RUN: not %lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/test-arm64.o %t/arm64e.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-ARM64
# RUN: %lld -arch arm64 -platform_version macos 13.0 13.0 %t/test-arm64.o \
# RUN:   %t/arm64.o -o /dev/null

## arm64e.x1 objects have the arm64e ABI but need a newer CPU than arm64e.
# RUN: llvm-mc -filetype=obj -triple=arm64e.x1-apple-macos -o %t/arm64e-x1.o \
# RUN:   %t/lib.s
# RUN: not %no-fatal-warnings-lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   %t/test-arm64.o %t/arm64e-x1.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARN-X1-ARM64
# RUN: not %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   %t/test.o %t/arm64e-x1.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=WARN-X1-ARM64E

# WARN: warning: {{.*}}arm64.o has architecture arm64 which is incompatible with target architecture arm64e
# WARN: error: undefined symbol: _helper
# ERR:  error: {{.*}}arm64.o has architecture arm64 which is incompatible with target architecture arm64e

# WARN-ARM64: warning: {{.*}}arm64e.o has architecture arm64e which is incompatible with target architecture arm64
# WARN-ARM64: error: undefined symbol: _helper
# ERR-ARM64:  error: {{.*}}arm64e.o has architecture arm64e which is incompatible with target architecture arm64

# WARN-X1-ARM64:  warning: {{.*}}arm64e-x1.o has architecture arm64e.x1 which is incompatible with target architecture arm64
# WARN-X1-ARM64E: warning: {{.*}}arm64e-x1.o has architecture arm64e.x1 which is incompatible with target architecture arm64e

#--- test.s
.globl _main
.p2align 2
_main:
  bl _helper
  ret

#--- lib.s
.globl _helper
.p2align 2
_helper:
  ret
