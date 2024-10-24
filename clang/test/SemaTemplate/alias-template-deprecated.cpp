// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
//
// This test checks that a deprecated attribute on an alias
// template triggers a warning diagnostic when it is used.

template <typename T>
struct NoAttr {
  void foo() {}
};

// expected-note@+2 7{{'UsingWithAttr' has been explicitly marked deprecated here}}
template <typename T>
using UsingWithAttr __attribute__((deprecated)) = NoAttr<T>;

// expected-note@+1 {{'UsingInstWithAttr' has been explicitly marked deprecated here}}
using UsingInstWithAttr __attribute__((deprecated)) = NoAttr<int>;

// expected-note@+1 {{'TDWithAttr' has been explicitly marked deprecated here}}
typedef NoAttr<int> TDWithAttr __attribute__((deprecated));

// expected-warning@+1 {{'UsingWithAttr' is deprecated}}
typedef UsingWithAttr<int> TDUsingWithAttr;

typedef NoAttr<int> TDNoAttr;

// expected-note@+1 {{'UsingTDWithAttr' has been explicitly marked deprecated here}}
using UsingTDWithAttr __attribute__((deprecated)) = TDNoAttr;

struct S {
  NoAttr<float> f1;
  // expected-warning@+1 {{'UsingWithAttr' is deprecated}}
  UsingWithAttr<float> f2;
};

// expected-warning@+1 {{'UsingWithAttr' is deprecated}}
void foo(NoAttr<short> s1, UsingWithAttr<short> s2) {
}

// expected-note@+2 {{'UsingWithCPPAttr' has been explicitly marked deprecated here}}
template <typename T>
using UsingWithCPPAttr [[deprecated]] = NoAttr<T>;

// expected-note@+1 {{'UsingInstWithCPPAttr' has been explicitly marked deprecated here}}
using UsingInstWithCPPAttr [[deprecated("Do not use this")]] = NoAttr<int>;

void bar() {
  NoAttr<int> obj; // Okay

  // expected-warning@+2 {{'UsingWithAttr' is deprecated}}
  // expected-note@+1 {{in instantiation of template type alias 'UsingWithAttr' requested here}}
  UsingWithAttr<int> objUsingWA;

  // expected-warning@+2 {{'UsingWithAttr' is deprecated}}
  // expected-note@+1 {{in instantiation of template type alias 'UsingWithAttr' requested here}}
  NoAttr<UsingWithAttr<int>> s;

  // expected-note@+1 {{'DepInt' has been explicitly marked deprecated here}}
  using DepInt [[deprecated]] = int;
  // expected-warning@+3 {{'UsingWithAttr' is deprecated}}
  // expected-warning@+2 {{'DepInt' is deprecated}}
  // expected-note@+1 {{in instantiation of template type alias 'UsingWithAttr' requested here}}
  using X = UsingWithAttr<DepInt>;

  // expected-warning@+2 {{'UsingWithAttr' is deprecated}}
  // expected-note@+1 {{in instantiation of template type alias 'UsingWithAttr' requested here}}
  UsingWithAttr<int>().foo();

  // expected-warning@+1 {{'UsingInstWithAttr' is deprecated}}
  UsingInstWithAttr objUIWA;

  // expected-warning@+1 {{'TDWithAttr' is deprecated}}
  TDWithAttr objTDWA;

  // expected-warning@+1 {{'UsingTDWithAttr' is deprecated}}
  UsingTDWithAttr objUTDWA;

  // expected-warning@+2 {{'UsingWithCPPAttr' is deprecated}}
  // expected-note@+1 {{in instantiation of template type alias 'UsingWithCPPAttr' requested here}}
  UsingWithCPPAttr<int> objUsingWCPPA;

  // expected-warning@+1 {{'UsingInstWithCPPAttr' is deprecated: Do not use this}}
  UsingInstWithCPPAttr objUICPPWA;
}
