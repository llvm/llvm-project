// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=X86
// RUN: %clang_cc1 -triple aarch64-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=AARCH64

// Test that __int256 works correctly with variadic functions (va_arg).

typedef __builtin_va_list va_list;

// x86_64: return via sret (Memory class per SysV ABI)
// X86-LABEL: define{{.*}} void @va_int256(ptr dead_on_unwind noalias writable sret(i256) align 16 %{{.*}}, i32 noundef %n, ...)
// X86: load i256, ptr %{{.*}}, align 16

// AArch64: return directly (4 GPRs)
// AARCH64-LABEL: define{{.*}} i256 @va_int256(i32 noundef %n, ...)
// AARCH64: load i256, ptr %{{.*}}, align
__int256 va_int256(int n, ...) {
  va_list ap;
  __builtin_va_start(ap, n);
  __int256 v = __builtin_va_arg(ap, __int256);
  __builtin_va_end(ap);
  return v;
}

// Test passing __int256 to a variadic function call.
void callee(int, ...);

// x86_64: __int256 passed via byval pointer
// X86-LABEL: define{{.*}} void @pass_int256(ptr noundef byval(i256) align 16 %0)
// X86: call void (i32, ...) @callee(i32 noundef 1, ptr noundef byval(i256) align 16 %

// AArch64: __int256 passed directly
// AARCH64-LABEL: define{{.*}} void @pass_int256(i256 noundef %x)
// AARCH64: call void (i32, ...) @callee(i32 noundef 1, i256 noundef %
void pass_int256(__int256 x) {
  callee(1, x);
}

// Multiple va_arg fetches of __int256
// X86-LABEL: define{{.*}} void @va_two(ptr{{.*}}sret(i256){{.*}}, i32 noundef %n, ...)
// X86: load i256, ptr %{{.*}}, align 16
// X86: load i256, ptr %{{.*}}, align 16
// X86: add nsw i256

// AARCH64-LABEL: define{{.*}} i256 @va_two(i32 noundef %n, ...)
// AARCH64: load i256
// AARCH64: load i256
// AARCH64: add nsw i256
__int256 va_two(int n, ...) {
  va_list ap;
  __builtin_va_start(ap, n);
  __int256 a = __builtin_va_arg(ap, __int256);
  __int256 b = __builtin_va_arg(ap, __int256);
  __builtin_va_end(ap);
  return a + b;
}

// Mixed sizes in varargs: int, __int256, long long
// X86-LABEL: define{{.*}} i64 @va_mixed(i32 noundef %n, ...)
// AARCH64-LABEL: define{{.*}} i64 @va_mixed(i32 noundef %n, ...)
long long va_mixed(int n, ...) {
  va_list ap;
  __builtin_va_start(ap, n);
  int x = __builtin_va_arg(ap, int);
  __int256 big = __builtin_va_arg(ap, __int256);
  long long y = __builtin_va_arg(ap, long long);
  __builtin_va_end(ap);
  return x + (long long)big + y;
}
