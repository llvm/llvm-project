// RUN: %clang_cc1 -std=c++1z -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

namespace std {
  using size_t = decltype(sizeof(0));
  template<typename> struct tuple_size;
  template<size_t, typename> struct tuple_element;
}

struct Y { int n; };
struct X { X(); X(Y); X(const X&); ~X(); };

struct A { int a : 13; bool b; };

struct B {};
template<> struct std::tuple_size<B> { enum { value = 2 }; };
template<> struct std::tuple_element<0,B> { using type = X; };
template<> struct std::tuple_element<1,B> { using type = const int&; };
template<int N> auto get(B) {
  if constexpr (N == 0)
    return Y();
  else
    return 0.0;
}

using C = int[2];

typedef int D __attribute__((ext_vector_type(2)));

using E = _Complex int;

template<typename T> T &make();

// CHECK: @_ZDC2a12a2E = global {{.*}} zeroinitializer, align 4
auto [a1, a2] = make<A>();
// CHECK: @_ZDC2b12b2E = global {{.*}} zeroinitializer, align 1
// CHECK: @b1 = global {{.*}}* null, align 8
// CHECK: @_ZGR2b1_ = internal global {{.*}} zeroinitializer, align 1
// CHECK: @b2 = global i32* null, align 8
// CHECK: @_ZGR2b2_ = internal global i32 0, align 4
auto [b1, b2] = make<B>();
// CHECK: @_ZDC2c12c2E = global [2 x i32]* null, align 8
auto &[c1, c2] = make<C>();
// CHECK: @_ZDC2d12d2E = global <2 x i32> zeroinitializer, align 8
auto [d1, d2] = make<D>();
// CHECK: @_ZDC2e12e2E = global { i32, i32 } zeroinitializer, align 4
auto [e1, e2] = make<E>();

// CHECK: call {{.*}}* @_Z4makeI1AERT_v()
// CHECK: call {{.*}}memcpy{{.*}}@_ZDC2a12a2E

// CHECK: @_Z4makeI1BERT_v()
//   CHECK: call i32 @_Z3getILi0EEDa1B()
//   CHECK: call void @_ZN1XC1E1Y({{.*}}* @_ZGR2b1_, i32
//   CHECK: call i32 @__cxa_atexit({{.*}}@_ZN1XD1Ev{{.*}}@_ZGR2b1_
//   CHECK: store {{.*}}* @_ZGR2b1_,
//
//   CHECK: call double @_Z3getILi1EEDa1B()
//   CHECK: fptosi double %{{.*}} to i32
//   CHECK: store i32 %{{.*}}, i32* @_ZGR2b2_
//   CHECK: store i32* @_ZGR2b2_, i32** @b2

// CHECK: call {{.*}}* @_Z4makeIA2_iERT_v()
// CHECK: store {{.*}}, [2 x i32]** @_ZDC2c12c2E

// CHECK: call {{.*}}* @_Z4makeIDv2_iERT_v()
// CHECK: store {{.*}}, <2 x i32>* @_ZDC2d12d2E, align 8

// CHECK: call {{.*}}* @_Z4makeICiERT_v()
// CHECK: store i32 %{{.*}}, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_ZDC2e12e2E, i32 0, i32 0)
// CHECK: store i32 %{{.*}}, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_ZDC2e12e2E, i32 0, i32 1)

// CHECK: define i32 @_Z12test_globalsv()
int test_globals() {
  return a2 + b2 + c2 + d2 + e2;
  // CHECK: load i8, i8* getelementptr inbounds (%struct.A, %struct.A* @_ZDC2a12a2E, i32 0, i32 1)
  //
  // CHECK: %[[b2:.*]] = load i32*, i32** @b2
  // CHECK: load i32, i32* %[[b2]]
  //
  // CHECK: %[[c1c2:.*]] = load [2 x i32]*, [2 x i32]** @_ZDC2c12c2E
  // CHECK: %[[c2:.*]] = getelementptr inbounds [2 x i32], [2 x i32]* %[[c1c2]], i64 0, i64 1
  // CHECK: load i32, i32* %[[c2]]
  //
  // CHECK: %[[d1d2:.*]] = load <2 x i32>, <2 x i32>* @_ZDC2d12d2E
  // CHECK: extractelement <2 x i32> %[[d1d2]], i32 1
  //
  // CHECK: load i32, i32* getelementptr inbounds ({ i32, i32 }, { i32, i32 }* @_ZDC2e12e2E, i32 0, i32 1)
}

// CHECK: define i32 @_Z11test_localsv()
int test_locals() {
  auto [b1, b2] = make<B>();

  // CHECK: @_Z4makeI1BERT_v()
  //   CHECK: call i32 @_Z3getILi0EEDa1B()
  //   CHECK: call void @_ZN1XC1E1Y({{.*}}* %[[b1:.*]], i32
  //
  //   CHECK: call double @_Z3getILi1EEDa1B()
  //   CHECK: %[[cvt:.*]] = fptosi double %{{.*}} to i32
  //   CHECK: store i32 %[[cvt]], i32* %[[b2:.*]],
  //   CHECK: store i32* %[[b2]], i32** %[[b2ref:.*]],

  return b2;
  // CHECK: %[[b2:.*]] = load i32*, i32** %[[b2ref]]
  // CHECK: load i32, i32* %[[b2]]

  // CHECK: call {{.*}}@_ZN1XD1Ev({{.*}}%[[b1]])
}

// CHECK: define void @_Z13test_bitfieldR1A(
void test_bitfield(A &a) {
  auto &[a1, a2] = a;
  a1 = 5;
  // CHECK: load i16, i16* %[[BITFIELD:.*]],
  // CHECK: and i16 %{{.*}}, -8192
  // CHECK: or i16 %{{.*}}, 5
  // CHECK: store i16 %{{.*}}, i16* %[[BITFIELD]],
}
