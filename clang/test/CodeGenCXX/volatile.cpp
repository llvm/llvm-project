// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -std=c++98 -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -std=c++11 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s

// Check that IR gen doesn't try to do an lvalue-to-rvalue conversion
// on a volatile reference result.  rdar://problem/8338198
namespace test0 {
  struct A {
    A(const A& t);
    A& operator=(const A& t);
    volatile A& operator=(const volatile A& t) volatile;
  };

  volatile A *array;

  // CHECK-LABEL: define void @_ZN5test04testENS_1AE(
  void test(A t) {
    // CHECK:      [[ARR:%.*]] = load [[A:%.*]]*, [[A:%.*]]** @_ZN5test05arrayE, align 8
    // CHECK-NEXT: [[IDX:%.*]] = getelementptr inbounds [[A]], [[A]]* [[ARR]], i64 0
    // CHECK-NEXT: [[TMP:%.*]] = call dereferenceable({{[0-9]+}}) [[A]]* @_ZNV5test01AaSERVKS0_([[A]]* [[IDX]], [[A]]* dereferenceable({{[0-9]+}}) [[T:%.*]])
    // CHECK-NEXT: ret void
    array[0] = t;
  }
}

namespace test1 {
  volatile int *x;

  // CHECK-LABEL: define void @_ZN5test14testEv()
  void test() {
    // CHECK:      [[TMP:%.*]] = load i32*, i32** @_ZN5test11xE, align 8
    // CHECK11-NEXT: {{%.*}} = load volatile i32, i32* [[TMP]], align 4
    // CHECK-NEXT: ret void
    *x;
  }
}
