// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t1.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t1.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t1.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t1.ll %s

template<typename T>
struct implicitly_instantiated {
  static T member;
};

template<typename T>
T implicitly_instantiated<T>::member = 12345;

int use_implicitly_instantiated() {
  return implicitly_instantiated<int>::member;
}

// CHECK-DAG: cir.global linkonce_odr comdat @_ZN23implicitly_instantiatedIiE6memberE = #cir.int<12345> : !s32i
// LLVM-DAG: @_ZN23implicitly_instantiatedIiE6memberE = linkonce_odr global i32 12345, comdat
// OGCG-DAG: @_ZN23implicitly_instantiatedIiE6memberE = linkonce_odr global i32 12345, comdat

template<typename T>
struct explicitly_instantiated {
    static T member;
};

template<typename T>
T explicitly_instantiated<T>::member = 54321;

template int explicitly_instantiated<int>::member;
// CHECK-DAG: cir.global weak_odr comdat @_ZN23explicitly_instantiatedIiE6memberE = #cir.int<54321> : !s32i
// LLVM-DAG: @_ZN23explicitly_instantiatedIiE6memberE = weak_odr global i32 54321, comdat
// OGCG-DAG: @_ZN23explicitly_instantiatedIiE6memberE = weak_odr global i32 54321, comdat

void func1(void) {
  // Should lower default-initialized static vars.
  static int i;
  // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func1vE1i = #cir.int<0> : !s32i

  // Should lower constant-initialized static vars.
  static int j = 1;
  // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func1vE1j = #cir.int<1> : !s32i

  // Should properly shadow static vars in nested scopes.
  {
    static int j = 2;
    // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func1vE1j_0 = #cir.int<2> : !s32i
  }
  {
    static int j = 3;
    // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func1vE1j_1 = #cir.int<3> : !s32i
  }

  // Should lower basic static vars arithmetics.
  j++;
  // CHECK-DAG: %[[#V2:]] = cir.get_global @_ZZ5func1vE1j : !cir.ptr<!s32i>
  // CHECK-DAG: %[[#V3:]] = cir.load{{.*}} %[[#V2]] : !cir.ptr<!s32i>, !s32i
  // CHECK-DAG: %[[#V4:]] = cir.unary(inc, %[[#V3]]) nsw : !s32i, !s32i
  // CHECK-DAG: cir.store{{.*}} %[[#V4]], %[[#V2]] : !s32i, !cir.ptr<!s32i>
}

// Should shadow static vars on different functions.
void func2(void) {
  static char i;
  // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func2vE1i = #cir.int<0> : !s8i
  static float j;
  // CHECK-DAG: cir.global "private" internal dso_local @_ZZ5func2vE1j = #cir.fp<0.000000e+00> : !cir.float
}

// CHECK-DAG: cir.global linkonce_odr comdat @_ZZ4testvE1c = #cir.int<0> : !s32i

// LLVM-DAG: $_ZZ4testvE1c = comdat any
// LLVM-DAG: @_ZZ4testvE1c = linkonce_odr global i32 0, comdat, align 4
// OGCG-DAG: $_ZZ4testvE1c = comdat any
// OGCG-DAG: @_ZZ4testvE1c = linkonce_odr global i32 0, comdat, align 4

inline void test() { static int c; }
// CHECK-LABEL: @_Z4testv
// CHECK: {{%.*}} = cir.get_global @_ZZ4testvE1c : !cir.ptr<!s32i>
void foo() { test(); }
