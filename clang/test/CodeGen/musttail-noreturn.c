// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -emit-llvm %s -o - | FileCheck %s

// A musttail call to a noreturn function must still be followed by a return.

[[noreturn]] void std_noreturn(int);
__attribute__((noreturn)) void gnu_noreturn(int);
typedef void noreturn_fn(int) __attribute__((noreturn));
extern noreturn_fn *noreturn_ptr;

// CHECK-LABEL: define {{.*}} @test_std(
// CHECK: musttail call void @std_noreturn(
// CHECK-NEXT: ret void
void test_std(int x) { [[clang::musttail]] return std_noreturn(x); }

// CHECK-LABEL: define {{.*}} @test_gnu(
// CHECK: musttail call void @gnu_noreturn(
// CHECK-NEXT: ret void
void test_gnu(int x) { [[clang::musttail]] return gnu_noreturn(x); }

// CHECK-LABEL: define {{.*}} @test_ptr(
// CHECK: musttail call void %{{.*}}(
// CHECK-NEXT: ret void
void test_ptr(int x) { [[clang::musttail]] return noreturn_ptr(x); }

// swiftasynccall makes the call musttail implicitly.
__attribute__((swiftasynccall, noreturn)) void
async_noreturn(char *__attribute__((swift_async_context)) ctx);

// CHECK-LABEL: define {{.*}} @test_swiftasync(
// CHECK: musttail call swifttailcc void @async_noreturn(
// CHECK-NEXT: ret void
__attribute__((swiftasynccall)) void
test_swiftasync(char *__attribute__((swift_async_context)) ctx) {
  return async_noreturn(ctx);
}
