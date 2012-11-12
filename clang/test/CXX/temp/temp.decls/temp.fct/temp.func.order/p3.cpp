// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace DeduceVsMember {
  template<typename T>
  struct X {
    template<typename U>
    int &operator==(const U& other) const;
  };

  template<typename T, typename U>
  float &operator==(const T&, const X<U>&);

  void test(X<int> xi, X<float> xf) {
    float& ir = (xi == xf);
  }
}

namespace OrderWithStaticMember {
  struct A {
    template<class T> int g(T**, int=0) { return 0; }
    template<class T> static int g(T*) { return 1; }
  };
  void f() {
    A a;
    int **p;
    a.g(p);
  }
}
