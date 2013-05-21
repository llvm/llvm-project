// RUN: %clang_cc1 -triple i386-pc-win32 %s -emit-llvm -fms-compatibility -o - | FileCheck %s

struct __declspec(align(16)) S {
  char x;
};
union { struct S s; } u;

// CHECK: @u = {{.*}}zeroinitializer, align 16


// CHECK: define void @t3() [[NAKED:#[0-9]+]] {
__declspec(naked) void t3() {}

// CHECK: define void @t22() [[NUW:#[0-9]+]]
void __declspec(nothrow) t22();
void t22() {}

// CHECK: define void @t2() [[NI:#[0-9]+]] {
__declspec(noinline) void t2() {}

// CHECK: call void @f20_t() [[NR:#[0-9]+]]
__declspec(noreturn) void f20_t(void);
void f20(void) { f20_t(); }

// CHECK: attributes [[NAKED]] = { naked noinline nounwind{{.*}} }
// CHECK: attributes [[NUW]] = { nounwind{{.*}} }
// CHECK: attributes [[NI]] = { noinline nounwind{{.*}} }
// CHECK: attributes [[NR]] = { noreturn }
