// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -fms-compatibility -O2 -disable-llvm-passes -o - | FileCheck %s

__declspec(selectany) int x1 = 1;
const __declspec(selectany) int x2 = 2;
// CHECK: @x1 = weak_odr dso_local global i32 1, comdat, align 4
// CHECK: @x2 = weak_odr dso_local constant i32 2, comdat, align 4

// selectany turns extern variable declarations into definitions.
__declspec(selectany) int x3;
extern __declspec(selectany) int x4;
// CHECK: @x3 = weak_odr dso_local global i32 0, comdat, align 4
// CHECK: @x4 = weak_odr dso_local global i32 0, comdat, align 4

struct __declspec(align(16)) S {
  char x;
};
union { struct S s; } u;

// CHECK: @u = {{.*}}zeroinitializer, align 16


// CHECK: define dso_local void @t3() [[NAKED:#[0-9]+]] {
__declspec(naked) void t3(void) {}

// CHECK: define dso_local void @t22() [[NUW:#[0-9]+]]
void __declspec(nothrow) t22(void);
void t22(void) {}

// CHECK: define dso_local void @t2() [[NI:#[0-9]+]] {
__declspec(noinline) void t2(void) {}

// CHECK: call void @f20_t() [[NR:#[0-9]+]]
__declspec(noreturn) void f20_t(void);
void f20(void) { f20_t(); }

__declspec(noalias) void noalias_callee(int *x);
// CHECK: call void @noalias_callee({{.*}}) [[NA:#[0-9]+]]
void noalias_caller(int *x) { noalias_callee(x); }

// CHECK: attributes [[NAKED]] = { naked noinline nounwind{{.*}} }
// CHECK: attributes [[NUW]] = { nounwind{{.*}} }
// CHECK: attributes [[NI]] = { noinline nounwind{{.*}} }
// CHECK: attributes [[NR]] = { noreturn }
// CHECK: attributes [[NA]] = { argmemonly nounwind{{.*}} }
