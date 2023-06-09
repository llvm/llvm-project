// RUN: %clang_cc1 -std=c++11 -S -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s
// RUN: %clang_cc1 -std=c++17 -S -emit-llvm -o - %s -triple x86_64-linux-gnu | FileCheck %s

struct A { int a, b; int f(); };

namespace NonAggregateCopyInAggregateInit { // PR32044
  struct A { constexpr A(int n) : x(n), y() {} int x, y; } extern a;
  // CHECK-DAG: @_ZN31NonAggregateCopyInAggregateInit1bE ={{.*}} global %{{.*}} { ptr @_ZN31NonAggregateCopyInAggregateInit1aE }
  struct B { A &p; } b{{a}};
  // CHECK-DAG: @_ZGRN31NonAggregateCopyInAggregateInit1cE_ = internal global %[[A:.*]] { i32 1, i32 0 }
  // CHECK-DAG: @_ZN31NonAggregateCopyInAggregateInit1cE ={{.*}} global %{{.*}} { ptr @_ZGRN31NonAggregateCopyInAggregateInit1cE_ }
  struct C { A &&p; } c{{1}};
}

namespace NearlyZeroInit {
  // CHECK-DAG: @_ZN14NearlyZeroInit1aE ={{.*}} global {{.*}} <{ i32 1, i32 2, i32 3, [120 x i32] zeroinitializer }>
  int a[123] = {1, 2, 3};
  // CHECK-DAG: @_ZN14NearlyZeroInit1bE ={{.*}} global {{.*}} { i32 1, <{ i32, [2147483647 x i32] }> <{ i32 2, [2147483647 x i32] zeroinitializer }> }
  struct B { int n; int arr[1024 * 1024 * 1024 * 2u]; } b = {1, {2}};
}

namespace PR37560 {
  union U {
    char x;
    int a;
  };
  // FIXME: [dcl.init]p2, the padding bits of the union object should be
  // initialized to 0, not undef, which would allow us to collapse the tail
  // of these arrays to zeroinitializer.
  // CHECK-DAG: @_ZN7PR375601cE ={{.*}} global <{ { i8, [3 x i8] } }> <{ { i8, [3 x i8] } { i8 0, [3 x i8] undef } }>
  U c[1] = {};
  // CHECK-DAG: @_ZN7PR375601dE ={{.*}} global {{.*}} <{ { i8, [3 x i8] } { i8 97, [3 x i8] undef }, %"{{[^"]*}}" { i32 123 }, { i8, [3 x i8] } { i8 98, [3 x i8] undef }, { i8, [3 x i8] } { i8 0, [3 x i8] undef },
  U d[16] = {'a', {.a = 123}, 'b'};
  // CHECK-DAG: @_ZN7PR375601eE ={{.*}} global {{.*}} <{ %"{{[^"]*}}" { i32 123 }, %"{{[^"]*}}" { i32 456 }, { i8, [3 x i8] } { i8 0, [3 x i8] undef },
  U e[16] = {{.a = 123}, {.a = 456}};

  union V {
    int a;
    char x;
  };
  // CHECK-DAG: @_ZN7PR375601fE ={{.*}} global [1 x %"{{[^"]*}}"] zeroinitializer
  V f[1] = {};
  // CHECK-DAG: @_ZN7PR375601gE ={{.*}} global {{.*}} <{ { i8, [3 x i8] } { i8 97, [3 x i8] undef }, %"{{[^"]*}}" { i32 123 }, { i8, [3 x i8] } { i8 98, [3 x i8] undef }, [13 x %"{{[^"]*}}"] zeroinitializer }>
  V g[16] = {{.x = 'a'}, {.a = 123}, {.x = 'b'}};
  // CHECK-DAG: @_ZN7PR375601hE ={{.*}} global {{.*}} <{ %"{{[^"]*}}" { i32 123 }, %"{{[^"]*}}" { i32 456 }, [14 x %"{{[^"]*}}"] zeroinitializer }>
  V h[16] = {{.a = 123}, {.a = 456}};
  // CHECK-DAG: @_ZN7PR375601iE ={{.*}} global [4 x %"{{[^"]*}}"] [%"{{[^"]*}}" { i32 123 }, %"{{[^"]*}}" { i32 456 }, %"{{[^"]*}}" zeroinitializer, %"{{[^"]*}}" zeroinitializer]
  V i[4] = {{.a = 123}, {.a = 456}};
}

// CHECK-LABEL: define {{.*}}@_Z3fn1i(
int fn1(int x) {
  // CHECK: %[[INITLIST:.*]] = alloca %struct.A
  // CHECK: %[[A:.*]] = getelementptr inbounds %struct.A, ptr %[[INITLIST]], i32 0, i32 0
  // CHECK: store i32 %{{.*}}, ptr %[[A]], align 4
  // CHECK: %[[B:.*]] = getelementptr inbounds %struct.A, ptr %[[INITLIST]], i32 0, i32 1
  // CHECK: store i32 5, ptr %[[B]], align 4
  // CHECK: call noundef i32 @_ZN1A1fEv(ptr {{[^,]*}} %[[INITLIST]])
  return A{x, 5}.f();
}

struct B { int &r; int &f() { return r; } };

// CHECK-LABEL: define {{.*}}@_Z3fn2Ri(
int &fn2(int &v) {
  // CHECK: %[[INITLIST2:.*]] = alloca %struct.B, align 8
  // CHECK: %[[R:.*]] = getelementptr inbounds %struct.B, ptr %[[INITLIST2:.*]], i32 0, i32 0
  // CHECK: store ptr %{{.*}}, ptr %[[R]], align 8
  // CHECK: call noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN1B1fEv(ptr {{[^,]*}} %[[INITLIST2:.*]])
  return B{v}.f();
}

// CHECK-LABEL: define {{.*}}@__cxx_global_var_init(
//
// CHECK: call {{.*}}@_ZN14NonTrivialInit1AC1Ev(
// CHECK: getelementptr inbounds {{.*}}, i64 1
// CHECK: br i1
//
// CHECK: getelementptr inbounds {{.*}}, i64 1
// CHECK: icmp eq {{.*}}, getelementptr inbounds {{.*}}, i64 30
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
  char data_8[1000 * 1000 * 1000] = {};
  int (&&data_9)[1000 * 1000 * 1000] = {0};
  unsigned char data_10[1024 * 1024 * 1024 * 2u] = { 1 };
  unsigned char data_11[1024 * 1024 * 1024 * 2u] = { One };
  unsigned char data_12[1024][1024][1024] = {{{1}}};

  // This variable must be initialized elementwise.
  Filler data_e1[1024] = {};
  // CHECK: getelementptr inbounds {{.*}} @_ZN8ZeroInit7data_e1E

  struct Largeish {
    long a, b, c;
  };
  // CHECK: define {{.*}}@_ZN8ZeroInit9largeish1Ev(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Largeish largeish1() { return {}; }
  // CHECK: define {{.*}}@_ZN8ZeroInit9largeish2Ev(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Largeish largeish2() { return Largeish(); }
  // CHECK: define {{.*}}@_ZN8ZeroInit9largeish3Ev(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Largeish largeish3() { return Largeish{}; }
  // CHECK: define {{.*}}@_ZN8ZeroInit9largeish4Ev(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Largeish largeish4() { return (Largeish){}; }
  // CHECK: define {{.*}}@_ZN8ZeroInit9largeish5Ev(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Largeish largeish5() { return {0, 0, 0}; }

  typedef __attribute__((ext_vector_type(4))) char CI4;
  struct Conversions {
    _Complex int a;
    _Complex float b;
    short c;
    long double d;
    CI4 e;
    char f;
    char g;
    int *h;
    long i;
  };
  // CHECK: define {{.*}}@_ZN8ZeroInit11conversionsEv(
  // CHECK-NOT }
  // CHECK: call {{.*}}memset
  Conversions conversions() {
    return {0,
            0,
            0,
            0,
            CI4(0),
            static_cast<char>(0.0),
            char(0 + 0i),
            reinterpret_cast<int *>(0),
            reinterpret_cast<long>((int *)nullptr)};
  }
}
