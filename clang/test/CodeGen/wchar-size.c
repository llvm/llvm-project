// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -emit-llvm -o - %s | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang_cc1 -triple s390x-none-zos -emit-llvm -o - %s | FileCheck %s -check-prefix=DEFAULT
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -o - -fwchar-type=short -fno-signed-wchar %s | FileCheck %s -check-prefix=OVERRIDE
// The wchar_size module flag is only emitted when the effective wchar_t size
// differs from the target triple's default.

// DEFAULT-NOT: wchar_size
// OVERRIDE: !{{[0-9]+}} = !{i32 {{[0-9]+}}, !"wchar_size", i32 2}
