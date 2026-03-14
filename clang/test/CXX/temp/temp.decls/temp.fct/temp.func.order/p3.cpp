// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// expected-no-diagnostics

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

#if __cplusplus >= 201103L
namespace OperatorWithRefQualifier {
  struct A { };
  template<class T> struct B {
    template<class R> int &operator*(R&) &&;
  };

  template<class T, class R> float &operator*(T&&, R&);
  void test() {
    A a;
    B<A> b;
    float &ir = b * a;
    int &ir2 = B<A>() * a;
  }
}

namespace PR17075 {
  template <typename T> struct V {};
  struct S { template<typename T> S &operator>>(T &t) = delete; };
  template<typename T> S &operator>>(S &s, V<T> &v);
  void f(S s, V<int> v) { s >> v; }
}
#endif
