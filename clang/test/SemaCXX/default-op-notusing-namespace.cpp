// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

struct X {};
namespace NS {
  bool operator==(X, X);
}

struct Y {
  X x;
  friend bool operator==(Y, Y);
};

// There is no `using namespace NS;` here, so this operator==
// will be implicitly deleted due to missing viable operator== for X.
bool operator==(Y, Y) = default;
// expected-error@-1 {{defaulting this equality comparison operator would delete it after its first declaration}}
// expected-note@9 {{defaulted 'operator==' is implicitly deleted because there is no viable 'operator==' for member 'x'}}
