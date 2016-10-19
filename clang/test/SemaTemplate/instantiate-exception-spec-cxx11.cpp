// RUN: %clang_cc1 -fsyntax-only -verify -triple %itanium_abi_triple -std=c++11 -ftemplate-depth 16 -fcxx-exceptions -fexceptions %s

// DR1330: an exception specification for a function template is only
// instantiated when it is needed.

// Note: the test is Itanium-specific because it depends on key functions in the
// PR12763 namespace.

template<typename T> void f1(T*) throw(T); // expected-error{{incomplete type 'Incomplete' is not allowed in exception specification}}
struct Incomplete; // expected-note{{forward}}

void test_f1(Incomplete *incomplete_p, int *int_p) {
  f1(int_p);
  f1(incomplete_p); // expected-note{{instantiation of exception spec}}
}

template<typename T> struct A {
  template<typename U> struct B {
    static void f() noexcept(A<U>().n);
  };

  constexpr A() : n(true) {}
  bool n;
};

static_assert(noexcept(A<int>::B<char>::f()), "");

template<unsigned N> struct S {
  static void recurse() noexcept(noexcept(S<N+1>::recurse())); // \
  // expected-error {{no member named 'recurse'}} \
  // expected-note 9{{instantiation of exception spec}}
};

template<> struct S<10> {};
void (*pFn2)() noexcept = &S<0>::recurse; // expected-note {{instantiation of exception spec}} expected-error {{not superset}}


namespace dr1330_example {
  template <class T> struct A {
    void f(...) throw (typename T::X);
    void f(int);
  };

  int main() {
    A<int>().f(42);
  }

  struct S {
    template<typename T> static void f() throw(typename T::X);
    typedef decltype(f<S>()) X; // expected-error {{exception specification is not available}}
  };

  struct T {
    template<typename T> static void f() throw(T**);
    typedef decltype(f<S>()) X; // expected-error {{exception specification is not available}}
  };

  template<typename T>
  struct U {
    void f() noexcept(T::error);
    void (g)() noexcept(T::error);
  };
  U<int> uint; // ok
}

namespace core_19754_example {
  template<typename T> T declval() noexcept;

  template<typename T, typename = decltype(T(declval<T&&>()))>
  struct is_movable { static const bool value = true; };

  template<typename T>
  struct wrap {
    T val;
    void irrelevant(wrap &p) noexcept(is_movable<T>::value);
  };

  template<typename T>
  struct base {
     base() {}
     base(const typename T::type1 &);
     base(const typename T::type2 &);
  };

  template<typename T>
  struct type1 {
     wrap<typename T::base> base;
  };

  template<typename T>
  struct type2 {
     wrap<typename T::base> base;
  };

  struct types {
     typedef base<types> base;
     typedef type1<types> type1;
     typedef type2<types> type2;
  };

  base<types> val = base<types>();
}

namespace pr9485 {
  template <typename T> void f1(T) throw(typename T::exception); // expected-note {{candidate}}
  template <typename T> void f1(T, int = 0) throw(typename T::noitpecxe); // expected-note {{candidate}}

  template <typename T> void f2(T) noexcept(T::throws); // expected-note {{candidate}}
  template <typename T> void f2(T, int = 0) noexcept(T::sworht); // expected-note {{candidate}}

  void test() {
    f1(0); // expected-error {{ambiguous}}
    f2(0); // expected-error {{ambiguous}}
  }
}

struct Exc1 { char c[4]; };
struct Exc2 { double x, y, z; };
struct Base {
  virtual void f() noexcept; // expected-note {{overridden}}
};
template<typename T> struct Derived : Base {
  void f() noexcept (sizeof(T) == 4); // expected-error {{is more lax}}
  void g() noexcept (T::error);
};

Derived<Exc1> d1; // ok
Derived<Exc2> d2; // expected-note {{in instantiation of}}

// If the vtable for a derived class is used, the exception specification of
// any member function which ends up in that vtable is needed, even if it was
// declared in a base class.
namespace PR12763 {
  template<bool *B> struct T {
    virtual void f() noexcept (*B); // expected-error {{constant expression}} expected-note {{read of non-const}}
  };
  bool b; // expected-note {{here}}
  struct X : public T<&b> {
    virtual void g();
  };
  void X::g() {} // expected-note {{in instantiation of}}
}

namespace Variadic {
  template<bool B> void check() { static_assert(B, ""); }
  template<bool B, bool B2, bool ...Bs> void check() { static_assert(B, ""); check<B2, Bs...>(); }

  template<typename ...T> void consume(T...);

  template<typename ...T> void f(void (*...p)() throw (T)) {
    void (*q[])() = { p... };
    consume((p(),0)...);
  }
  template<bool ...B> void g(void (*...p)() noexcept (B)) {
    consume((p(),0)...);
    check<noexcept(p()) == B ...>();
  }
  template<typename ...T> void i() {
    consume([]() throw(T) {} ...);
    consume([]() noexcept(sizeof(T) == 4) {} ...);
  }
  template<bool ...B> void j() {
    consume([](void (*p)() noexcept(B)) {
      void (*q)() noexcept = p; // expected-error {{not superset of source}}
    } ...);
  }

  void z() {
    f<int, char, double>(nullptr, nullptr, nullptr);
    g<true, false, true>(nullptr, nullptr, nullptr);
    i<int, long, short>();
    j<true, true>();
    j<true, false>(); // expected-note {{in instantiation of}}
  }

}

namespace NondefDecls {
  template<typename T> void f1() {
    int g1(int) noexcept(T::error); // expected-error{{type 'int' cannot be used prior to '::' because it has no members}}
  }
  template void f1<int>(); // expected-note{{in instantiation of function template specialization 'NondefDecls::f1<int>' requested here}}
}

