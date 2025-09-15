// RUN: %clang_cl --target=aarch64-windows-msvc -Xclang -verify /E -U__STDC_HOSTED__ -Wno-builtin-macro-redefined %s 2>&1 | FileCheck %s

// expected-no-diagnostics

// CHECK: void __yield(void);
#include <intrin.h>
void f() { __yield(); }

