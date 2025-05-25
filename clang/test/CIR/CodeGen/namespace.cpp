// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - 2>&1 | FileCheck %s

// Test anonymous namespace.
namespace {
  int g1 = 1;

  void f1(void) {}
}


// Test named namespace.
namespace test {
  int g2 = 2;
  void f2(void);

  // Test nested namespace.
  namespace test2 {
    int g3 = 3;
    void f3(void);
  }
}

// CHECK-DAG: cir.global internal @_ZN12_GLOBAL__N_12g1E = #cir.int<1> : !s32i
// CHECK-DAG: cir.global external @_ZN4test2g2E = #cir.int<2> : !s32i
// CHECK-DAG: cir.global external @_ZN4test5test22g3E = #cir.int<3> : !s32i
// CHECK-DAG: cir.func @_ZN12_GLOBAL__N_12f1Ev()
// CHECK-DAG: cir.func @_ZN4test2f2Ev()
// CHECK-DAG: cir.func @_ZN4test5test22f3Ev()

using namespace test;

// Test global function.
int f4(void) {
    f1();
    f2();
    test2::f3();
    return g1 + g2 + test2::g3;
}

// The namespace gets added during name mangling, so this is wrong but expected.
// CHECK: cir.func @_Z2f4v()
// CHECK:   cir.call @_ZN12_GLOBAL__N_12f1Ev()
// CHECK:   cir.call @_ZN4test2f2Ev()
// CHECK:   cir.call @_ZN4test5test22f3Ev()
// CHECK:   %[[G1_ADDR:.*]] = cir.get_global @_ZN12_GLOBAL__N_12g1E : !cir.ptr<!s32i>
// CHECK:   %[[G1_VAL:.*]] = cir.load{{.*}} %[[G1_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[G2_ADDR:.*]] = cir.get_global @_ZN4test2g2E : !cir.ptr<!s32i>
// CHECK:   %[[G2_VAL:.*]] = cir.load{{.*}} %[[G2_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[SUM:.*]] = cir.binop(add, %[[G1_VAL]], %[[G2_VAL]]) nsw : !s32i
// CHECK:   %[[G3_ADDR:.*]] = cir.get_global @_ZN4test5test22g3E : !cir.ptr<!s32i>
// CHECK:   %[[G3_VAL:.*]] = cir.load{{.*}} %[[G3_ADDR]] : !cir.ptr<!s32i>, !s32i
// CHECK:   %[[SUM2:.*]] = cir.binop(add, %[[SUM]], %[[G3_VAL]]) nsw : !s32i
