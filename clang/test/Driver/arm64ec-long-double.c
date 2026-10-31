// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-64 -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LD64
// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-80 -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LD80
// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-128 -c %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=LD128
// RUN: not %clang --target=arm64ec-pc-windows-msvc -mlong-double-64 \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=MSVC
// RUN: not %clang --target=aarch64-w64-mingw32 -mlong-double-80 \
// RUN:   -fsyntax-only %s 2>&1 | FileCheck %s --check-prefix=AARCH64

// ARM64EC with a non-MSVC ABI defaults to the x87 80-bit format of x86_64, so
// all three widths are accepted and forwarded to the frontend.
// LD64: "-mlong-double-64"
// LD80: "-mlong-double-80"
// LD128: "-mlong-double-128"

// The MSVC ABI keeps its binary64 `long double` for compatibility, and plain
// AArch64 has no reason to support the x87 format at all.
// MSVC: error: unsupported option '-mlong-double-64' for target 'arm64ec-pc-windows-msvc{{.*}}'
// AARCH64: error: unsupported option '-mlong-double-80' for target 'aarch64-w64-windows-gnu'
