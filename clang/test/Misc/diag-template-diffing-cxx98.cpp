// RUN: not %clang_cc1 -fsyntax-only %s -std=c++98 2>&1 | FileCheck %s

namespace PR14342 {
  template<typename T, char a> struct X {};
  X<int, 1> x = X<long, 257>();
  // CHECK: error: no viable conversion from 'X<long, [...]>' to 'X<int, [...]>'
}

namespace PR15513 {
  template <int x, int y = x+1>
  class A {};

  void foo(A<0> &M) {
    // CHECK: no viable conversion from 'A<[...], (default) x + 1 aka 1>' to 'A<[...], 0>'
    A<0, 0> N = M;
   // CHECK: no viable conversion from 'A<0, [...]>' to 'A<1, [...]>'
    A<1, 1> O = M;
  }
}

namespace default_args {
  template <int x, int y = 1+1, int z = 2>
  class A {};

  void foo(A<0> &M) {
    // CHECK: no viable conversion from 'A<[...], (default) 1 + 1 aka 2, (default) 2>' to 'A<[...], 0, 0>'
    A<0, 0, 0> N = M;

    // CHECK: no viable conversion from 'A<[2 * ...], (default) 2>' to 'A<[2 * ...], 0>'
    A<0, 2, 0> N2 = M;
  }

}

namespace qualifiers {
  template <class T>
  void foo(void (func(T*)), T*) {}

  template <class T>
  class vector{};

  void bar(const vector<int>*) {}

  void test(volatile vector<int>* V) {
    foo(bar, V);
  }

  // CHECK: candidate template ignored: deduced conflicting types for parameter 'T' ('const vector<...>' vs. 'volatile vector<...>')
}

namespace integers {
  template <int x>
  class wrapper{};

  template <int x>
  class foo {
   public:
    wrapper<x> make();
  };

  wrapper<1> w1 = foo<2>().make();
  // CHECK: no viable conversion from 'wrapper<2>' to 'wrapper<1>'

  wrapper<1> w2 = foo<-3>().make();
  // CHECK: no viable conversion from 'wrapper<-3>' to 'wrapper<1>'

  template <int x>
  wrapper<x> make();

  wrapper<1> w3 = make<4>();
  // CHECK: no viable conversion from 'wrapper<4>' to 'wrapper<1>'

  template <int x>
  wrapper<-x> makeNegative();

  wrapper<1> w4 = makeNegative<5>();
  // CHECK: no viable conversion from 'wrapper<-5>' to 'wrapper<1>'

  wrapper<1> w5 = makeNegative<-6>();
  // CHECK: no viable conversion from 'wrapper<6>' to 'wrapper<1>'
}
