// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

using size_t = decltype(sizeof(int));

struct S {
  constexpr int f(); // expected-warning {{C++14}}
  constexpr int g() const;
  constexpr int h(); // expected-warning {{C++14}}
  int h();
  static constexpr int Sf();
  /*static*/ constexpr void *operator new(size_t) noexcept;
  template<typename T> constexpr T tm(); // expected-warning {{C++14}}
  template<typename T> static constexpr T ts();
};

void f(const S &s) {
  s.f();
  s.g();

  int (*Sf)() = &S::Sf;
  int (S::*f)() const = &S::f;
  int (S::*g)() const = &S::g;
  void *(*opNew)(size_t) = &S::operator new;
  int (S::*tm)() const = &S::tm;
  int (*ts)() = &S::ts;
}

constexpr int S::f() const { return 0; }
constexpr int S::g() { return 1; } // expected-warning {{C++14}}
constexpr int S::h() { return 0; } // expected-warning {{C++14}}
int S::h() { return 0; }
constexpr int S::Sf() { return 2; }
constexpr void *S::operator new(size_t) noexcept { return 0; }
template<typename T> constexpr T S::tm() { return T(); } // expected-warning {{C++14}}
template<typename T> constexpr T S::ts() { return T(); }

namespace std_example {

  class debug_flag {
  public:
    explicit debug_flag(bool);
    constexpr bool is_on() const; // ok (dr1684)
  private:
    bool flag;
  };

  constexpr int bar(int x, int y) // expected-note {{here}}
    { return x + y + x*y; }
  int bar(int x, int y) // expected-error {{non-constexpr declaration of 'bar' follows constexpr declaration}}
    { return x * 2 + 3 * y; }

}

// The constexpr specifier is allowed for static member functions of non-literal types.
class NonLiteralClass {
  NonLiteralClass(bool);
  static constexpr bool isDebugFlag();
};
