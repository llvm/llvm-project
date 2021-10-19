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
  auto g1() noexcept(noexcept(f(ptr))) -> decltype(f(ptr));
  auto g2() const noexcept(noexcept(f(((this))->ptr))) -> decltype(f(ptr));
  auto g3() noexcept(noexcept(f(this->ptr))) -> decltype(f((*this).ptr));
  auto g4() const noexcept(noexcept(f(((this))->ptr))) -> decltype(f(this->ptr));
  auto g5() noexcept(noexcept(this->f(ptr))) -> decltype(this->f(ptr));
  auto g6() const noexcept(noexcept(this->f(((this))->ptr))) -> decltype(this->f(ptr));
  auto g7() noexcept(noexcept(this->f(this->ptr))) -> decltype(this->f((*this).ptr));
  auto g8() const noexcept(noexcept(this->f(((this))->ptr))) -> decltype(this->f(this->ptr));
};

void test_C(C<int> ci) {
  int &ir = ci.g1();
  float &fr = ci.g2();
  int &ir2 = ci.g3();
  float &fr2 = ci.g4();
  int &ir3 = ci.g5();
  float &fr3 = ci.g6();
  int &ir4 = ci.g7();
  float &fr4 = ci.g8();
  static_assert(!noexcept(ci.g1()), "exception-specification failure");
  static_assert(noexcept(ci.g2()), "exception-specification failure");
  static_assert(!noexcept(ci.g3()), "exception-specification failure");
  static_assert(noexcept(ci.g4()), "exception-specification failure");
  static_assert(!noexcept(ci.g5()), "exception-specification failure");
  static_assert(noexcept(ci.g6()), "exception-specification failure");
  static_assert(!noexcept(ci.g7()), "exception-specification failure");
  static_assert(noexcept(ci.g8()), "exception-specification failure");
}

namespace PR14263 {
  template<typename T> struct X {
    void f();
    T f() const;

    auto g()       -> decltype(this->f()) { return f(); }
    auto g() const -> decltype(this->f()) { return f(); }
  };
  template struct X<int>;
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

namespace PR15290 {
  template<typename T>
  class A {
    T v_;
    friend int add_to_v(A &t) noexcept(noexcept(v_ + 42))
    {
      return t.v_ + 42;
    }
  };
  void f()
  {
    A<int> t;
    add_to_v(t);
  }
}

namespace Static {
  struct X1 {
    int m;
    // FIXME: This should be accepted.
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
    void bar(Base&) {}
  };

  struct Derived : Base {
    void foo(Derived& d) noexcept(noexcept(d.bar(d))) {}
  };
}

namespace rdar13473493 {
  template <typename F>
  class wrap
  {
  public:
    template <typename... Args>
    auto operator()(Args&&... args) const -> decltype(wrapped(args...)) // expected-note{{candidate template ignored: substitution failure [with Args = <int>]: member 'wrapped' used before its declaration}}
    {
      return wrapped(args...);
    }
  
  private:
    F wrapped;
  };

  void test(wrap<int (*)(int)> w) {
    w(5); // expected-error{{no matching function for call to object of type 'wrap<int (*)(int)>'}}
  }
}
