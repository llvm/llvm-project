// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

namespace PR15360 {
  template<typename R, typename U, R F>
  U f() { return &F; } // expected-error{{cannot take the address of an rvalue of type 'int (*)(int)'}} expected-error{{cannot take the address of an rvalue of type 'int *'}}
  void test() {
    f<int(int), int(*)(int), nullptr>(); // expected-note{{in instantiation of}}
    f<int[3], int*, nullptr>(); // expected-note{{in instantiation of}}
  }
}

namespace CanonicalNullptr {
  template<typename T> struct get { typedef T type; };
  struct X {};
  template<typename T, typename get<T *>::type P = nullptr> struct A {};
  template<typename T, typename get<decltype((T(), nullptr))>::type P = nullptr> struct B {};
  template<typename T, typename get<T X::*>::type P = nullptr> struct C {};

  template<typename T> A<T> MakeA();
  template<typename T> B<T> MakeB();
  template<typename T> C<T> MakeC();
  A<int> a = MakeA<int>();
  B<int> b = MakeB<int>();
  C<int> c = MakeC<int>();
}

namespace Auto {
  template<auto> struct A { };  // expected-error {{until C++17}}
}

namespace check_conversion_early {
  struct X {};
  template<int> struct A {};
  template<X &x> struct A<x> {}; // expected-error {{not implicitly convertible}}

  struct Y { constexpr operator int() const { return 0; } };
  template<Y &y> struct A<y> {}; // expected-error {{cannot be deduced}} expected-note {{'y'}}
}

namespace ReportCorrectParam {
template <int a, unsigned b, int c>
void TempFunc() {}

void Useage() {
  //expected-error@+2 {{no matching function}}
  //expected-note@-4 {{candidate template ignored: invalid explicitly-specified argument for template parameter 'b'}}
  TempFunc<1, -1, 1>();
}
}

namespace PR42513 {
  template<typename X, int Ret = WidgetCtor((X*)nullptr)> void f();
  constexpr int WidgetCtor(struct X1*);

  struct X1 {
    friend constexpr int WidgetCtor(X1*);
  };
  template<typename X1>
  struct StandardWidget {
    friend constexpr int WidgetCtor(X1*) {
      return 0;
    }
  };
  template struct StandardWidget<X1>;

  void use() { f<X1>(); }
}

namespace ReferenceToConstexpr {
  struct A { const char *str = "hello"; };
  constexpr A a;
  template<const A &r, typename T> struct B {
    static_assert(__builtin_strcmp(r.str, "hello") == 0, "");
  };
  template<const A &r> struct C {
    template<typename T> void f(B<r, T>, T) {}
  };
  void f(C<a> ca) { ca.f({}, 0); }
}
