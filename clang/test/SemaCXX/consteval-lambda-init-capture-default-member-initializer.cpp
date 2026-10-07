// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
// RUN: %clang_cc1 -std=c++23 -fsyntax-only -verify %s

int gv = 5; // expected-note 2 {{declared here}}
constexpr int cgv = 5;

struct W {
  int value;
  consteval W(int x) : value(x) {}
};

consteval int id(int x) { return x; }

struct S {
  int member = [captured = W(gv).value] { return captured; }(); // expected-error {{call to consteval function 'W::W' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct F {
  int member = [captured = id(gv)] { return captured; }(); // expected-error {{call to consteval function 'id' is not a constant expression}} \
  // expected-note {{declared here}} \
  // expected-note {{read of non-const variable 'gv' is not allowed in a constant expression}}
};

struct Valid {
  int member = [captured = W(cgv).value] { return captured; }();
  int function = [captured = id(cgv)] { return captured; }();
};

static_assert(Valid{}.member == 5);
static_assert(Valid{}.function == 5);

int main() {
  return S{}.member + F{}.member; // expected-note 2 {{in the default initializer of 'member'}}
}
