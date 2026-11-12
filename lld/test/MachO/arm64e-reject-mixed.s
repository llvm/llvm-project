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

## Dylibs and bitcode are checked too. Like ld-prime, an arm64 link takes an
## arm64e dylib.
# RUN: %lld -arch arm64 -platform_version macos 13.0 13.0 -dylib %t/arm64.o \
# RUN:   -o %t/libarm64.dylib
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -dylib %t/arm64e.o \
# RUN:   -o %t/libarm64e.dylib
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/test.o %t/libarm64.dylib -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-DYLIB
# RUN: %lld -arch arm64 -platform_version macos 13.0 13.0 -arch_errors_fatal \
# RUN:   %t/test-arm64.o %t/libarm64e.dylib -o /dev/null

## Bitcode is checked in either input order, so LTO never compiles arm64 bitcode
## into an arm64e image. -ObjC doesn't load an ignored lazy file either.
# RUN: llvm-as %t/main.ll -o %t/main.bc
# RUN: llvm-as %t/lib.ll -o %t/lib.bc
# RUN: opt -mtriple=arm64-apple-macos13.0 %t/lib.ll -o %t/lib-arm64.bc
# RUN: opt -mtriple=arm64-apple-macos13.0 %t/objc.ll -o %t/objc-arm64.bc
# RUN: llvm-ar rcs %t/lib-arm64.a %t/lib-arm64.bc
# RUN: %lld -arch arm64e -platform_version macos 13.0 13.0 -arch_errors_fatal \
# RUN:   %t/main.bc %t/lib.bc -o /dev/null
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/main.bc %t/lib-arm64.bc -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-BC
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/lib-arm64.bc %t/main.bc -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-BC
# RUN: not %lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/main.bc %t/lib-arm64.a -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-BC
# RUN: not %lld -arch arm64 -platform_version macos 13.0 13.0 \
# RUN:   -arch_errors_fatal %t/test-arm64.o %t/lib.bc -o /dev/null 2>&1 | \
# RUN:   FileCheck %s --check-prefix=ERR-BC-ARM64
# RUN: not %no-fatal-warnings-lld -arch arm64e -platform_version macos 13.0 13.0 \
# RUN:   -ObjC %t/test.o --start-lib %t/objc-arm64.bc --end-lib -o /dev/null \
# RUN:   2>&1 | FileCheck %s --check-prefix=WARN-OBJC

# ERR-DYLIB:    error: {{.*}}libarm64.dylib has architecture arm64 which is incompatible with target architecture arm64e
# ERR-BC:       error: {{.*}}lib-arm64.{{.*}} has architecture arm64 which is incompatible with target architecture arm64e
# ERR-BC-ARM64: error: {{.*}}lib.bc has architecture arm64e which is incompatible with target architecture arm64
# WARN-OBJC:    warning: {{.*}}objc-arm64.bc has architecture arm64 which is incompatible with target architecture arm64e
# WARN-OBJC:    error: undefined symbol: _helper

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

#--- main.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-macos13.0"

declare void @helper()

define void @main() {
  call void @helper()
  ret void
}

#--- lib.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-macos13.0"

define void @helper() {
  ret void
}

#--- objc.ll
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64e-apple-macos13.0"

@category = private global i8 0
@catlist = private global [1 x ptr] [ptr @category], section "__DATA,__objc_catlist,regular,no_dead_strip"

define void @helper() {
  ret void
}
