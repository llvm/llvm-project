// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -std=c++98 -o - | FileCheck -check-prefix=CHECK -check-prefix=CHECK98 %s
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

  // CHECK-LABEL: define{{.*}} void @_ZN5test04testENS_1AE(
  void test(A t) {
    // CHECK:      [[ARR:%.*]] = load ptr, ptr @_ZN5test05arrayE, align 8
    // CHECK-NEXT: [[IDX:%.*]] = getelementptr inbounds [[A:%.*]], ptr [[ARR]], i64 0
    // CHECK-NEXT: [[TMP:%.*]] = call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZNV5test01AaSERVKS0_(ptr {{[^,]*}} [[IDX]], ptr noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[T:%.*]])
    // CHECK-NEXT: ret void
    array[0] = t;
  }
}

namespace test1 {
  volatile int *x;

  // CHECK-LABEL: define{{.*}} void @_ZN5test14testEv()
  void test() {
    // CHECK:      [[TMP:%.*]] = load ptr, ptr @_ZN5test11xE, align 8
    // CHECK11-NEXT: {{%.*}} = load volatile i32, ptr [[TMP]], align 4
    // CHECK-NEXT: ret void
    *x;
  }
}

namespace PR40642 {
  template <class T> struct S {
    // CHECK-LABEL: define {{.*}} @_ZN7PR406421SIiE3fooEv(
    void foo() {
      // CHECK98-NOT: load volatile
      // CHECK11: load volatile
      if (true)
        reinterpret_cast<const volatile unsigned char *>(m_ptr)[0];
      // CHECK: }
    }
    int *m_ptr;
  };

  void f(S<int> *x) { x->foo(); }
}
