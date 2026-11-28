// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-64 -c %s 2>&1 | FileCheck %s --check-prefix=LD64
// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-80 -c %s 2>&1 | FileCheck %s --check-prefix=LD80
// RUN: %clang -### --target=arm64ec-w64-mingw32 -mlong-double-128 -c %s 2>&1 | FileCheck %s --check-prefix=LD128
// RUN: not %clang -### --target=arm64ec-pc-windows-msvc -mlong-double-80 -c %s 2>&1 | FileCheck %s --check-prefix=MSVC
// RUN: not %clang -### --target=aarch64-w64-mingw32 -mlong-double-80 -c %s 2>&1 | FileCheck %s --check-prefix=AARCH64

// LD64: "-mlong-double-64"
// LD80: "-mlong-double-80"
// LD128: "-mlong-double-128"
// MSVC: error: unsupported option '-mlong-double-80' for target 'arm64ec-pc-windows-msvc{{.*}}'
// AARCH64: error: unsupported option '-mlong-double-80' for target 'aarch64-w64-windows-gnu'
