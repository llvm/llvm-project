// RUN: %clang_cc1 -triple aarch64-unknown-uefi -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-pc-windows-msvc -emit-llvm -o - %s | FileCheck %s

// aarch64-unknown-uefi follows the Windows ARM64 ABI: LLP64 types, char* based
// va_list and the Microsoft C++ ABI.

#include <stdarg.h>

// CHECK: @sizeof_long = {{.*}}global i32 4
// CHECK: @sizeof_wchar = {{.*}}global i32 2
// CHECK: @sizeof_long_double = {{.*}}global i32 8
// CHECK: @sizeof_va_list = {{.*}}global i32 8
int sizeof_long = sizeof(long);
int sizeof_wchar = sizeof(__WCHAR_TYPE__);
int sizeof_long_double = sizeof(long double);
int sizeof_va_list = sizeof(va_list);

// char is signed on AArch64 Windows and UEFI.
// CHECK: @char_is_signed = {{.*}}global i32 1
int char_is_signed = (char)-1 < 0;

// CHECK-LABEL: define dso_local i32 @sum(i32 noundef %n, ...)
// CHECK: %ap = alloca ptr
// CHECK: call void @llvm.va_start.p0(ptr %ap)
// CHECK: load ptr, ptr %ap
// CHECK: getelementptr inbounds i8, ptr %{{.*}}, i64 8
int sum(int n, ...) {
  va_list ap;
  va_start(ap, n);
  int s = 0;
  for (int i = 0; i < n; i++)
    s += va_arg(ap, int);
  va_end(ap);
  return s;
}

// Small aggregates larger than 8 bytes are passed indirectly as in the
// Windows ARM64 ABI.
struct larger {
  int a, b, c, d, e;
};
// CHECK-LABEL: define dso_local void @pass_larger(ptr {{.*}}dead_on_return{{.*}} %s)
void pass_larger(struct larger s) {}
