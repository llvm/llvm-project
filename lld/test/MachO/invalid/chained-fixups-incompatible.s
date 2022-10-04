# REQUIRES: x86, aarch64
# RUN: llvm-mc -filetype=obj -triple=x86_64-apple-darwin %s -o %t.o
# RUN: not %lld -lSystem -no_pie -fixup_chains %t.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=NO-PIE
# RUN: %no-fatal-warnings-lld -fixup_chains %t.o -o /dev/null \
# RUN:   -lSystem -platform_version macos 10.15 10.15 2>&1 | \
# RUN:   FileCheck %s --check-prefix=VERSION
# RUN: llvm-mc -filetype=obj -triple=arm64_32-apple-darwin %s -o %t-arm64_32.o
# RUN: not %lld-watchos -lSystem -fixup_chains %t-arm64_32.o -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ARCH

## Check that we emit diagnostics when -fixup_chains is explicitly specified,
## but we don't support creating chained fixups for said configuration.
# NO-PIE:  error: -fixup_chains is incompatible with -no_pie
# VERSION: warning: -fixup_chains requires macOS 11.0, which is newer than target minimum of 10.15
# ARCH:    error: -fixup_chains is only supported on x86_64 and arm64 targets

.globl _main
_main:
  ret
