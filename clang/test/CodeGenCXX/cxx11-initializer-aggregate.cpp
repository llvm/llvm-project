// RUN: %clang_cc1 -std=c++11 -S -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s

struct A { int a, b; int f(); };

namespace NonAggregateCopyInAggregateInit { // PR32044
  struct A { constexpr A(int n) : x(n), y() {} int x, y; } extern a;
  // CHECK-DAG: @_ZN31NonAggregateCopyInAggregateInit1bE = global %{{.*}} { %[[A:.*]]* @_ZN31NonAggregateCopyInAggregateInit1aE }
  struct B { A &p; } b{{a}};
  // CHECK-DAG: @_ZGRN31NonAggregateCopyInAggregateInit1cE_ = internal global %[[A]] { i32 1, i32 0 }
  // CHECK-DAG: @_ZN31NonAggregateCopyInAggregateInit1cE = global %{{.*}} { %{{.*}}* @_ZGRN31NonAggregateCopyInAggregateInit1cE_ }
  struct C { A &&p; } c{{1}};
}

// CHECK-LABEL: define {{.*}}@_Z3fn1i(
int fn1(int x) {
  // CHECK: %[[INITLIST:.*]] = alloca %struct.A
  // CHECK: %[[A:.*]] = getelementptr inbounds %struct.A, %struct.A* %[[INITLIST]], i32 0, i32 0
  // CHECK: store i32 %{{.*}}, i32* %[[A]], align 4
  // CHECK: %[[B:.*]] = getelementptr inbounds %struct.A, %struct.A* %[[INITLIST]], i32 0, i32 1
  // CHECK: store i32 5, i32* %[[B]], align 4
  // CHECK: call i32 @_ZN1A1fEv(%struct.A* %[[INITLIST]])
  return A{x, 5}.f();
}

struct B { int &r; int &f() { return r; } };

// CHECK-LABEL: define {{.*}}@_Z3fn2Ri(
int &fn2(int &v) {
  // CHECK: %[[INITLIST2:.*]] = alloca %struct.B, align 8
  // CHECK: %[[R:.*]] = getelementptr inbounds %struct.B, %struct.B* %[[INITLIST2:.*]], i32 0, i32 0
  // CHECK: store i32* %{{.*}}, i32** %[[R]], align 8
  // CHECK: call dereferenceable({{[0-9]+}}) i32* @_ZN1B1fEv(%struct.B* %[[INITLIST2:.*]])
  return B{v}.f();
}

// CHECK-LABEL: define {{.*}}@__cxx_global_var_init(
//
// CHECK: call {{.*}}@_ZN14NonTrivialInit1AC1Ev(
// CHECK: getelementptr inbounds {{.*}}, i64 1
// CHECK: br i1
//
// CHECK: getelementptr inbounds {{.*}}, i64 1
// CHECK: icmp eq {{.*}}, i64 30
// CHECK: br i1
//
// CHECK: call i32 @__cxa_atexit(
namespace NonTrivialInit {
  struct A { A(); A(const A&) = delete; ~A(); };
  struct B { A a[20]; };
  // NB, this must be large enough to be worth memsetting for this test to be
  // meaningful.
  B b[30] = {};
}

namespace ZeroInit {
  enum { Zero, One };
  constexpr int zero() { return 0; }
  constexpr int *null() { return nullptr; }
  struct Filler {
    int x;
    Filler();
  };
  struct S1 {
    int x;
  };

  // These declarations, if implemented elementwise, require huge
  // amout of memory and compiler time.
  unsigned char data_1[1024 * 1024 * 1024 * 2u] = { 0 };
  unsigned char data_2[1024 * 1024 * 1024 * 2u] = { Zero };
  unsigned char data_3[1024][1024][1024] = {{{0}}};
  unsigned char data_4[1024 * 1024 * 1024 * 2u] = { zero() };
  int *data_5[1024 * 1024 * 512] = { nullptr };
  int *data_6[1024 * 1024 * 512] = { null() };
  struct S1 data_7[1024 * 1024 * 512] = {{0}};

  // This variable must be initialized elementwise.
  Filler data_e1[1024] = {};
  // CHECK: getelementptr inbounds {{.*}} @_ZN8ZeroInit7data_e1E
}
