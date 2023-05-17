// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++11 -fvisibility-inlines-hidden -emit-llvm -o - %s -O2 -disable-llvm-passes | FileCheck %s

// The trickery with optimization in the run line is to get IR
// generation to emit available_externally function bodies, but not
// actually inline them (and thus remove the emitted bodies).

struct X0 {
  void __attribute__((visibility("default"))) f1() { }
  void f2() { }
  void f3();
  static void f5() { }
  virtual void f6() { }
};

inline void X0::f3() { }

template<typename T>
struct X1 {
  void __attribute__((visibility("default"))) f1() { }
  void f2() { }
  void f3();
  void f4();
  static void f5() { }
  virtual void f6() { }
};

template<typename T>
inline void X1<T>::f3() { }

template<>
inline void X1<int>::f4() { }

struct __attribute__((visibility("default"))) X2 {
  void f2() { }
};

extern template struct X1<float>;

void use(X0 *x0, X1<int> *x1, X2 *x2, X1<float> *x3) {
  // CHECK-LABEL: define linkonce_odr void @_ZN2X02f1Ev
  x0->f1();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X02f2Ev
  x0->f2();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X02f3Ev
  x0->f3();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X02f5Ev
  X0::f5();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X02f6Ev
  x0->X0::f6();
  // CHECK-LABEL: define linkonce_odr void @_ZN2X1IiE2f1Ev
  x1->f1();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X1IiE2f2Ev
  x1->f2();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X1IiE2f3Ev
  x1->f3();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X1IiE2f4Ev
  x1->f4();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X1IiE2f5Ev
  X1<int>::f5();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X1IiE2f6Ev
  x1->X1::f6();
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN2X22f2Ev
  x2->f2();
  // CHECK-LABEL: define available_externally void @_ZN2X1IfE2f2Ev
  x3->f2();
}

// rdar://problem/8614470
namespace test1 {
  struct __attribute__((visibility("default"))) A {
    inline void foo();
    ~A();
  };

  void test() {
    A a;
    a.foo();
  }
// CHECK: declare void @_ZN5test11A3fooEv
// CHECK: declare {{.*}} @_ZN5test11AD1Ev
}

// PR8713
namespace test2 {
  struct A {};
  template <class T> class B {};
  typedef B<A> arg;

  namespace ns __attribute__((visibility("default"))) {
    template <class T> inline void foo() {}
    extern template void foo<arg>();
  }

  void test() {
    ns::foo<arg>();
  }

  // CHECK-LABEL: define available_externally void @_ZN5test22ns3fooINS_1BINS_1AEEEEEvv()
}

namespace PR11642 {
  template <typename T>
  class Foo {
  public:
    T foo(T x) { return x; }
  };
  extern template class Foo<int>;
  template class Foo<int>;
  // CHECK-LABEL: define weak_odr noundef i32 @_ZN7PR116423FooIiE3fooEi
}

// Test that clang implements the new gcc behaviour for inline functions.
// GCC PR30066.
namespace test3 {
  inline void foo(void) {
  }
  template<typename T>
  inline void zed() {
  }
  template void zed<float>();
  void bar(void) {
    foo();
    zed<int>();
  }
  // CHECK-LABEL: define weak_odr void @_ZN5test33zedIfEEvv
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN5test33fooEv
  // CHECK-LABEL: define linkonce_odr hidden void @_ZN5test33zedIiEEvv
}

namespace test4 {
  extern inline __attribute__ ((__gnu_inline__))
  void foo() {}
  void bar() {
    foo();
  }
  // CHECK-LABEL: define available_externally void @_ZN5test43fooE
}

namespace test5 {
  // just don't crash.
  template <int> inline void Op();
  class UnaryInstruction {
    UnaryInstruction() {
      Op<0>();
    }
  };
  template <int Idx_nocapture> void Op() {
  }
}

namespace test6 {
  // just don't crash.
  template <typename T>
  void f(T x) {
  }
  struct C {
    static void g() {
      f([](){});
    }
  };
  void g() {
    C::g();
  }
}

namespace PR34811 {
  template <typename T> void tf() {}
  
  // CHECK-LABEL: define linkonce_odr hidden noundef ptr @_ZN7PR348111fEv(
  inline void *f() {
    auto l = []() {};
    // CHECK-LABEL: define linkonce_odr hidden void @_ZN7PR348112tfIZNS_1fEvEUlvE_EEvv(
    return (void *)&tf<decltype(l)>;
  }
  
  void *p = (void *)f;
}
