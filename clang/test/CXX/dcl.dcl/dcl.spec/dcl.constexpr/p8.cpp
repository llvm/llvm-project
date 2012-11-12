// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct S {
  constexpr int f();
  constexpr int g() const;
  static constexpr int Sf();
};

void f(const S &s) {
  s.f();
  s.g();

  int (*f)() = &S::Sf;
  int (S::*g)() const = &S::g;
}

namespace std_example {

  class debug_flag { // expected-note {{not an aggregate and has no constexpr constructors}}
  public:
    explicit debug_flag(bool);
    constexpr bool is_on(); // expected-error {{non-literal type 'std_example::debug_flag' cannot have constexpr members}}
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
