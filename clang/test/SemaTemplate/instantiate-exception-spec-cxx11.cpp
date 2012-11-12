// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 -ftemplate-depth 16 -fcxx-exceptions -fexceptions %s

// DR1330: an exception specification for a function template is only
// instantiated when it is needed.

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
decltype(S<0>::recurse()) *pVoid1 = 0; // ok, exception spec not needed
decltype(&S<0>::recurse) pFn = 0; // ok, exception spec not needed

template<> struct S<10> {};
void (*pFn2)() noexcept = &S<0>::recurse; // expected-note {{instantiation of exception spec}} expected-error {{not superset}}


namespace dr1330_example {
  template <class T> struct A {
    void f(...) throw (typename T::X); // expected-error {{'int'}}
    void f(int);
  };

  int main() {
    A<int>().f(42);
  }

  int test2() {
    struct S {
      template<typename T>
      static int f() noexcept(noexcept(A<T>().f("boo!"))) { return 0; } // \
      // expected-note {{instantiation of exception spec}}
      typedef decltype(f<S>()) X;
    };
    S().f<S>(); // ok
    S().f<int>(); // expected-note {{instantiation of exception spec}}
  }
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
