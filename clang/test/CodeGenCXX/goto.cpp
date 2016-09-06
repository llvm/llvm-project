// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -fcxx-exceptions -fexceptions -emit-llvm -o - | FileCheck %s

// Reduced from a crash on boost::interprocess's node_allocator_test.cpp.
namespace test0 {
  struct A { A(); ~A(); };
  struct V { V(const A &a = A()); ~V(); };

  // CHECK-LABEL: define linkonce_odr i32 @_ZN5test04testILi0EEEii
  template<int X> int test(int x) {
    // CHECK:      [[RET:%.*]] = alloca i32
    // CHECK-NEXT: [[X:%.*]] = alloca i32
    // CHECK-NEXT: [[Y:%.*]] = alloca [[A:%.*]],
    // CHECK-NEXT: [[Z:%.*]] = alloca [[A]]
    // CHECK-NEXT: [[EXN:%.*]] = alloca i8*
    // CHECK-NEXT: [[SEL:%.*]] = alloca i32
    // CHECK-NEXT: [[V:%.*]] = alloca [[V:%.*]]*,
    // CHECK-NEXT: [[TMP:%.*]] = alloca [[A]]
    // CHECK-NEXT: [[CLEANUPACTIVE:%.*]] = alloca i1
    // CHECK:      call void @_ZN5test01AC1Ev([[A]]* [[Y]])
    // CHECK-NEXT: invoke void @_ZN5test01AC1Ev([[A]]* [[Z]])
    // CHECK:      [[NEW:%.*]] = invoke i8* @_Znwm(i64 1)
    // CHECK:      store i1 true, i1* [[CLEANUPACTIVE]]
    // CHECK:      [[NEWCAST:%.*]] = bitcast i8* [[NEW]] to [[V]]*
    // CHECK-NEXT: invoke void @_ZN5test01AC1Ev([[A]]* [[TMP]])
    // CHECK:      invoke void @_ZN5test01VC1ERKNS_1AE([[V]]* [[NEWCAST]], [[A]]* dereferenceable({{[0-9]+}}) [[TMP]])
    // CHECK:      store i1 false, i1* [[CLEANUPACTIVE]]
    // CHECK-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* [[TMP]])
    A y;
    try {
      A z;
      V *v = new V();

      if (x) return 1;
    } catch (int ex) {
      return 1;
    }
    return 0;
  }

  int test() {
    return test<0>(5);
  }
}
