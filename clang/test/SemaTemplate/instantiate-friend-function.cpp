// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -S -triple %itanium_abi_triple -std=c++11 -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

namespace PR10856 {
  template<typename T> class C;

  template<typename S, typename R = S> C<R> operator - (C<S> m0, C<S> m1);
  template<typename T> class C {
    public:
      template<typename S, typename R> friend C<R> operator - (C<S> m0, C<S> m1);
  };

  template<typename S, typename R> inline C<R> operator - (C<S> m0, C<S> m1) {
    C<R> ret;
    return ret;
  }
};

int PR10856_main(int argc, char** argv) {
  using namespace PR10856;
  C<int> a;
  a-a;
  return 0;
}

// PR10856::C<int> PR10856::operator-<int, int>(PR10856::C<int>, PR10856::C<int>)
// CHECK: define {{.*}} @_ZN7PR10856miIiiEENS_1CIT0_EENS1_IT_EES5_

namespace PR10856_Root {
  template<typename Value, typename Defaulted = void>
  bool g(Value value);

  template<typename ClassValue> class MyClass {
  private:
    template<typename Value, typename Defaulted>
    friend bool g(Value value);
  };
}

namespace PR10856_Root {
  void f() {
    MyClass<int> value;
    g(value);
  }
}

// bool PR10856_Root::g<PR10856_Root::MyClass<int>, void>(PR10856_Root::MyClass<int>)
// CHECK: call {{.*}} @_ZN12PR10856_Root1gINS_7MyClassIiEEvEEbT_

namespace PR43400 {
  template<typename T> struct X {
    friend void f() = delete;
  };
  X<int> xi;
}
