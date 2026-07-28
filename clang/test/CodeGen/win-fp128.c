// RUN: %clang_cc1 -triple x86_64-windows-gnu -emit-llvm -o - %s \
// RUN:    | FileCheck %s --check-prefix=CHECK-GNU64
// __float128 is unsupported on MSVC

__float128 fp128_ret(void) { return 0; }
// CHECK-GNU64: define dso_local void @fp128_ret(ptr dead_on_unwind noalias writable sret(fp128) align 16 %agg.result)

__float128 fp128_args(__float128 a, __float128 b) { return a * b; }
// CHECK-GNU64: define dso_local void @fp128_args(ptr dead_on_unwind noalias writable sret(fp128) align 16 %agg.result, ptr noundef align 16 dead_on_return %0, ptr noundef align 16 dead_on_return %1)

__float128 __attribute__((vectorcall)) fp128_ret_vectorcall(void) { return 0; }
// CHECK-GNU64: define dso_local x86_vectorcallcc fp128 @"\01fp128_ret_vectorcall@@0"()

__float128 __attribute__((regcall)) fp128_ret_regcall(void) { return 0; }
// CHECK-GNU64: define dso_local x86_regcallcc fp128 @__regcall3__fp128_ret_regcall()

__float128 fp128_callee(void);
__float128 fp128_musttail(void) { [[clang::musttail]] return fp128_callee(); }
// CHECK-GNU64: define dso_local void @fp128_musttail(ptr dead_on_unwind noalias writable sret(fp128) align 16 %agg.result)
// CHECK-GNU64: musttail call void @fp128_callee(ptr dead_on_unwind writable sret(fp128) align 16 %agg.result)

void fp128_vararg(int a, ...) {
  // CHECK-GNU64-LABEL: define dso_local void @fp128_vararg
  __builtin_va_list ap;
  __builtin_va_start(ap, a);
  __float128 i = __builtin_va_arg(ap, __float128);
  // CHECK-GNU64: load ptr, ptr
  // CHECK-GNU64: load fp128, ptr
  __builtin_va_end(ap);
}
