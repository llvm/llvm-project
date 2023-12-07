// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

// PR4678
namespace test0 {
  // test1 should be compmiled to be a varargs function in the IR even
  // though there is no way to do a va_begin.  Otherwise, the optimizer
  // will warn about 'dropped arguments' at the call site.

  // CHECK-LABEL: define{{.*}} i32 @_ZN5test05test1Ez(...)
  int test1(...) {
    return -1;
  }

  // CHECK: call noundef i32 (...) @_ZN5test05test1Ez(i32 noundef 0)
  void test() {
    test1(0);
  }
}

namespace test1 {
  struct A {
    int x;
    int y;
  };

  void foo(...);

  void test() {
    A x;
    foo(x);
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN5test14testEv()
  // CHECK:      [[X:%.*]] = alloca [[A:%.*]], align 4
  // CHECK-NEXT: [[TMP:%.*]] = alloca [[A]], align 4
  // CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 4 [[TMP]], ptr align 4 [[X]], i64 8, i1 false)
  // CHECK-NEXT: [[T1:%.*]] = load i64, ptr [[TMP]], align 4
  // CHECK-NEXT: call void (...) @_ZN5test13fooEz(i64 [[T1]])
  // CHECK-NEXT: ret void
}
