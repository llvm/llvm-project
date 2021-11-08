// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -emit-llvm %s -o - | FileCheck %s

// rdar://7309675
// PR4678
namespace test0 {
  // test1 should be compmiled to be a varargs function in the IR even
  // though there is no way to do a va_begin.  Otherwise, the optimizer
  // will warn about 'dropped arguments' at the call site.

  // CHECK-LABEL: define{{.*}} i32 @_ZN5test05test1Ez(...)
  int test1(...) {
    return -1;
  }

  // CHECK: call i32 (...) @_ZN5test05test1Ez(i32 0)
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
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[TMP]] to i8*
  // CHECK-NEXT: [[T1:%.*]] = bitcast [[A]]* [[X]] to i8*
  // CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 [[T0]], i8* align 4 [[T1]], i64 8, i1 false)
  // CHECK-NEXT: [[T0:%.*]] = bitcast [[A]]* [[TMP]] to i64*
  // CHECK-NEXT: [[T1:%.*]] = load i64, i64* [[T0]], align 4
  // CHECK-NEXT: call void (...) @_ZN5test13fooEz(i64 [[T1]])
  // CHECK-NEXT: ret void
}
