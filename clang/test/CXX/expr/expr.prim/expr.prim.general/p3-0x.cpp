// RUN: %clang_cc1 -fsyntax-only -std=c++11 %s -verify 

struct A {
  int &f(int*);
  float &f(int*) const noexcept;
  
  int *ptr;
  auto g1() noexcept(noexcept(f(ptr))) -> decltype(f(this->ptr));
  auto g2() const noexcept(noexcept(f((*this).ptr))) -> decltype(f(ptr));
};

void testA(A &a) {
  int &ir = a.g1();
  float &fr = a.g2();
  static_assert(!noexcept(a.g1()), "exception-specification failure");
  static_assert(noexcept(a.g2()), "exception-specification failure");
}

struct B {
  char g();
  template<class T> auto f(T t) -> decltype(t + g())
  { return t + g(); }
};

template auto B::f(int t) -> decltype(t + g());

template<typename T>
struct C {
  int &f(T*);
  float &f(T*) const noexcept;

  T* ptr;
  auto g1() noexcept(noexcept(f(ptr))) -> decltype(f((*this).ptr));
  auto g2() const noexcept(noexcept(f(((this))->ptr))) -> decltype(f(ptr));
};

void test_C(C<int> ci) {
  int *p = 0;
  int &ir = ci.g1();
  float &fr = ci.g2();
  static_assert(!noexcept(ci.g1()), "exception-specification failure");
  static_assert(noexcept(ci.g2()), "exception-specification failure");
}

namespace PR10036 {
  template <class I>
  void
  iter_swap(I x, I y) noexcept;

  template <class T>
  class A
  {
    T t_;
  public:
    void swap(A& a) noexcept(noexcept(iter_swap(&t_, &a.t_)));
  };

  void test() {
    A<int> i, j;
    i.swap(j);
  }
}

namespace Static {
  struct X1 {
    int m;
    static auto f() -> decltype(m); // expected-error{{'this' cannot be implicitly used in a static member function declaration}}
    static auto g() -> decltype(this->m); // expected-error{{'this' cannot be used in a static member function declaration}}

    static int h();
    
    static int i() noexcept(noexcept(m + 2)); // expected-error{{'this' cannot be implicitly used in a static member function declaration}}
  };

  auto X1::h() -> decltype(m) { return 0; } // expected-error{{'this' cannot be implicitly used in a static member function declaration}}

  template<typename T>
  struct X2 {
    int m;

    T f(T*);
    static T f(int);

    auto g(T x) -> decltype(f(x)) { return 0; }
  };

  void test_X2() {
    X2<int>().g(0);
  }
}

namespace PR12564 {
  struct Base {
    void bar(Base&) {} // FIXME: expected-note {{here}}
  };

  struct Derived : Base {
    // FIXME: This should be accepted.
    void foo(Derived& d) noexcept(noexcept(d.bar(d))) {} // expected-error {{cannot bind to a value of unrelated type}}
  };
}
