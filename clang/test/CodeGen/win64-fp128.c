// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=GNU64
// __float128 is unsupported on MSVC

__float128 fp128_ret(void) { return 0; }
// GNU64: define dso_local void @fp128_ret(ptr dead_on_unwind noalias writable sret(fp128) align 16 %agg.result)

__float128 fp128_args(__float128 a, __float128 b) { return a * b; }
// GNU64: define dso_local void @fp128_args(ptr dead_on_unwind noalias writable sret(fp128) align 16 %agg.result, ptr noundef %0, ptr noundef %1)

void fp128_vararg(int a, ...) {
  // GNU64-LABEL: define dso_local void @fp128_vararg
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  __float128 i = __builtin_va_arg(ap, __float128);
  // GNU64: load ptr, ptr
  // GNU64: load fp128, ptr
  __builtin_va_end(ap);
}
