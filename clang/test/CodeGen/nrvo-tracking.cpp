// RUN: %clang_cc1 -std=c++20 -fblocks -Wno-return-stack-address -triple x86_64-unknown-unknown-gnu -emit-llvm -O1 -fexperimental-new-pass-manager -o - %s | FileCheck %s

struct alignas(4) X {
  X();
  X(const X &);
  X(X &&);
};

#define L(A, B, C) void l##A() {    \
  auto t = []<class T = X>() -> C { \
    T t;                            \
    return B;                       \
  }();                              \
}

// CHECK-LABEL: define{{.*}} void @_Z2l1v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(1, t, X);

// CHECK-LABEL: define{{.*}} void @_Z2l2v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(2, t, X&);

// CHECK-LABEL: define{{.*}} void @_Z2l3v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(3, t, T);

// CHECK-LABEL: define{{.*}} void @_Z2l4v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(4, t, T&);

// CHECK-LABEL: define{{.*}} void @_Z2l5v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(5, t, auto);

// CHECK-LABEL: define{{.*}} void @_Z2l6v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(6, t, auto&);

// CHECK-LABEL: define{{.*}} void @_Z2l7v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(7, t, decltype(auto));

// CHECK-LABEL: define{{.*}} void @_Z2l8v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
L(8, (t), decltype(auto));

#undef L

#define F(A, B, C) template<class T = X> static inline auto tf##A() -> C { \
    T t;                                                                   \
    return B;                                                              \
}                                                                          \
void f##A() { auto t = tf##A(); }                                          \

// CHECK-LABEL: define{{.*}} void @_Z2f1v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(1, t, X);

// CHECK-LABEL: define{{.*}} void @_Z2f2v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(2, t, X&);

// CHECK-LABEL: define{{.*}} void @_Z2f3v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(3, t, T);

// CHECK-LABEL: define{{.*}} void @_Z2f4v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(4, t, T&);

// CHECK-LABEL: define{{.*}} void @_Z2f5v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(5, t, auto);

// CHECK-LABEL: define{{.*}} void @_Z2f6v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(6, t, auto&);

// CHECK-LABEL: define{{.*}} void @_Z2f7v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(7, t, decltype(auto));

// CHECK-LABEL: define{{.*}} void @_Z2f8v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
F(8, (t), decltype(auto));

#undef F

#define B(A, B) void b##A() { \
  auto t = []<class T = X>() { return ^ B () { \
      T t;                                     \
      return t;                                \
  }; }()();                                    \
}

// CHECK-LABEL: define{{.*}} void @_Z2b1v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
B(1, X);

// CHECK-LABEL: define{{.*}} void @_Z2b2v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
B(2, X&);

// CHECK-LABEL: define{{.*}} void @_Z2b3v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
B(3, T);

// CHECK-LABEL: define{{.*}} void @_Z2b4v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call {{.*}} @_ZN1XC1ERKS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
B(4, T&);

// CHECK-LABEL: define{{.*}} void @_Z2b5v
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
B(5, );

#undef B

// CHECK-LABEL: define{{.*}} void @_Z6f_attrv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
template<class T = X> [[gnu::cdecl]] static inline auto tf_attr() -> X {
  T t;
  return t;
}
void f_attr() { auto t = tf_attr(); }

// CHECK-LABEL: define{{.*}} void @_Z6b_attrv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call void @llvm.lifetime.end
// CHECK-NEXT:  ret void
void b_attr() {
  auto t = []<class T = X>() {
    return ^X() [[clang::vectorcall]] {
      T t;
      return t;
    };
  }()();
}

namespace test_alignas {

template <int A> X t1() {
  X a [[gnu::aligned(A)]];
  return a;
}

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t1ILi1EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  ret void
template X t1<1>();

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t1ILi4EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  ret void
template X t1<4>();

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t1ILi8EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
template X t1<8>();

template <int A> X t2() {
  X a [[gnu::aligned(1)]] [[gnu::aligned(A)]] [[gnu::aligned(2)]];
  return a;
}

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t2ILi1EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  ret void
template X t2<1>();

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t2ILi4EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  ret void
template X t2<4>();

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t2ILi8EEE1Xv
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
template X t2<8>();

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t3Ev
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  ret void
X t3() {
  X a [[gnu::aligned(1)]];
  return a;
}

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t4Ev
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
X t4() {
  X a [[gnu::aligned(8)]];
  return a;
}

// CHECK-LABEL: define{{.*}} void @_ZN12test_alignas2t5Ev
// CHECK:       call {{.*}} @_ZN1XC1Ev
// CHECK-NEXT:  call {{.*}} @_ZN1XC1EOS_
// CHECK-NEXT:  call void @llvm.lifetime.end
X t5() {
  X a [[gnu::aligned(1)]] [[gnu::aligned(8)]];
  return a;
}

} // namespace test_alignas

namespace PR51862 {

template <class T> T test() {
  T a;
  T b;
  if (0)
    return a;
  return b;
}

struct A {
  A();
  A(A &);
  A(int);
  operator int();
};

// CHECK-LABEL: define{{.*}} void @_ZN7PR518624testINS_1AEEET_v
// CHECK:       call noundef i32 @_ZN7PR518621AcviEv
// CHECK-NEXT:  call void @_ZN7PR518621AC1Ei
// CHECK-NEXT:  call void @llvm.lifetime.end
template A test<A>();

struct BSub {};
struct B : BSub {
  B();
  B(B &);
  B(const BSub &);
};

// CHECK-LABEL: define{{.*}} void @_ZN7PR518624testINS_1BEEET_v
// CHECK:       call void @_ZN7PR518621BC1ERKNS_4BSubE
// CHECK-NEXT:  call void @llvm.lifetime.end
template B test<B>();

} // namespace PR51862
