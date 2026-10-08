/// Tests that va_list layout matches the host target's getBuiltinVaListKind().

// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=LINUX %s
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -aux-triple x86_64-pc-windows-msvc \
// RUN:   -fsycl-is-device -emit-llvm -o - %s | FileCheck --check-prefix=WINDOWS %s

[[clang::sycl_external]] int f(int n, ...) {
  __builtin_va_list ap1, ap2;
  __builtin_va_start(ap1, n);
  int v = __builtin_va_arg(ap1, int);
  __builtin_va_copy(ap2, ap1);
  __builtin_va_end(ap1);
  __builtin_va_end(ap2);
  return v;
}

// LINUX:    define {{.*}} i32 @_Z1fiz(i32 noundef %n, ...) {{.*}} {
// LINUX:      %ap1 = alloca [1 x %struct.__va_list_tag], align 8
// LINUX:      %ap2 = alloca [1 x %struct.__va_list_tag], align 8

// WINDOWS:  define {{.*}} i32 @_Z1fiz(i32 noundef %n, ...) {{.*}} {
// WINDOWS:    %ap1 = alloca ptr addrspace(4), align 8
// WINDOWS:    %ap2 = alloca ptr addrspace(4), align 8
