// RUN: %clang_cc1 -no-opaque-pointers %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

void target() noexcept
{
  // CHECK: invoke void @_Z8externalv()
  external();
}
// CHECK:      [[T0:%.*]] = landingpad { i8*, i32 }
// CHECK-NEXT:  catch i8* null
// CHECK-NEXT: [[T1:%.*]] = extractvalue { i8*, i32 } [[T0]], 0
// CHECK-NEXT: call void @__clang_call_terminate(i8* [[T1]]) [[NR_NUW:#[0-9]+]]
// CHECK-NEXT: unreachable

void reverse() noexcept(false)
{
  // CHECK: call void @_Z8externalv()
  external();
}

// CHECK: attributes [[NR_NUW]] = { noreturn nounwind }
