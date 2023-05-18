// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

struct X { virtual ~X(); };
struct Y : public X { };
struct Z; // expected-note{{forward declaration of 'Z'}}

void test(X &x, Y &y, Z &z) {
  // If T is an rvalue reference type, v shall be an expression having
  // a complete class type, and the result is an xvalue of the type
  // referred to by T.
  Y &&yr0 = dynamic_cast<Y&&>(x);
  Y &&yr1 = dynamic_cast<Y&&>(static_cast<X&&>(x));
  Y &&yr2 = dynamic_cast<Y&&>(z); // expected-error{{'Z' is an incomplete type}}
}
