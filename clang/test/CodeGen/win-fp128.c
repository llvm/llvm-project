// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=CHECK-GNU64
// __float128 is unsupported on MSVC

__float128 fp128_ret(void) { return 0; }
// CHECK-GNU64: define dso_local <2 x i64>  @fp128_ret()

__float128 fp128_args(__float128 a, __float128 b) { return a * b; }
// CHECK-GNU64: define dso_local <2 x i64> @fp128_args(ptr noundef dead_on_return %0, ptr noundef dead_on_return %1)

void fp128_vararg(int a, ...) {
  // CHECK-GNU64-LABEL: define dso_local void @fp128_vararg
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  __float128 i = __builtin_va_arg(ap, __float128);
  // CHECK-GNU64: load ptr, ptr
  // CHECK-GNU64: load fp128, ptr
  __builtin_va_end(ap);
}
