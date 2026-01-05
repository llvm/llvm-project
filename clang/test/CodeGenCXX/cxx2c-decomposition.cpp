// RUN: %clang_cc1 -std=c++26 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

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

template<typename T> T &make();

// CHECK-LABEL: define {{.*}} @_Z8big_testIiEiv()
template <typename T>
int big_test() {
  A& a = make<A>();
A:
  auto &[...an] = a;
  an...[0] = 5;
  // CHECK: %[[a1:.*]].load = load i16, ptr %[[BITFIELD:.*]],
  // CHECK: %[[a1]].clear = and i16 %[[a1]].load, -8192
  // CHECK: %[[a1]].set = or i16 %[[a1]].clear, 5
  // CHECK: store i16 %[[a1]].set, ptr %[[BITFIELD]],
B:
  auto [b1, ...bn] = make<B>();
  // CHECK: @_Z4makeI1BERT_v()
  //   CHECK: call i32 @_Z3getILi0EEDa1B()
  //   CHECK: call void @_ZN1XC1E1Y(ptr {{[^,]*}} %[[b1:.*]], i32
  //
  //   CHECK: call noundef double @_Z3getILi1EEDa1B()
  //   CHECK: %[[cvt:.*]] = fptosi double %{{.*}} to i32
  //   CHECK: store i32 %[[cvt]], ptr %[[b2:.*]],
  //   CHECK: store ptr %[[b2]], ptr %[[b2ref:.*]],
  int bn2 = bn...[0];
  // CHECK load ptr, ptr %[[b2ref]]

  return 0;
}

int g = big_test<int>();
