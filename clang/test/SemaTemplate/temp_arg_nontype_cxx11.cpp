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

using FourChars = const char[4];
constexpr FourChars kEta = "Eta";
constexpr const char kDes[4] = "Des";
constexpr const char *kNull = "Phi";
constexpr const char **kZero[] = {};

template <const char *, typename T> class Column {};
template <const char[], typename T> class Dolumn {};
template <const char (*)[4], typename T> class Folumn {};
template <FourChars *, typename T> class Golumn {};
template <const char *const *, typename T> class Holumn {};
template <const char *const *const *, typename T> class Jolumn {};
template <const char **const (*)[0], typename T> class Iolumn {};

class container {
public:
  int a;
};
template <int container::*> class Kolumn {};

void lookup() {
  Column<kEta, double>().ls();    // expected-error {{<kEta,}}
  Column<kDes, double>().ls();    // expected-error {{<kDes,}}
  Column<nullptr, double>().ls(); // expected-error {{<nullptr,}}
  Dolumn<kEta, double>().ls();    // expected-error {{<kEta,}}
  Dolumn<kDes, double>().ls();    // expected-error {{<kDes,}}
  Folumn<&kEta, double>().ls();   // expected-error {{<&kEta,}}
  Folumn<&kDes, double>().ls();   // expected-error {{<&kDes,}}
  Golumn<&kEta, double>().ls();   // expected-error {{<&kEta,}}
  Golumn<&kDes, double>().ls();   // expected-error {{<&kDes,}}
  Holumn<&kNull, double>().ls();  // expected-error {{<&kNull,}}
  Jolumn<kZero, double>().ls();   // expected-error {{<kZero,}}
  Iolumn<&kZero, double>().ls();  // expected-error {{<&kZero,}}
  Kolumn<&container::a>().ls();   // expected-error {{<&container::a}}
  Kolumn<nullptr>().ls();         // expected-error {{<nullptr}}
}
