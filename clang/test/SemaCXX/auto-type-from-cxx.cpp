// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

struct A {
    operator __auto_type() {} // expected-error {{'__auto_type' not allowed in conversion function type}}
};

__auto_type a() -> int; // expected-error {{function with trailing return type must specify return type 'auto'}}
__auto_type a2(); // expected-error {{'__auto_type' not allowed in function return type}}
template <typename T>
__auto_type b() { return T::x; } // expected-error {{'__auto_type' not allowed in function return type}}
auto c() -> __auto_type { __builtin_unreachable(); } // expected-error {{'__auto_type' not allowed in function return type}}
int d() {
  decltype(__auto_type) e = 1; // expected-error {{expected expression}}
  auto _ = [](__auto_type f) {}; // expected-error {{'__auto_type' not allowed in lambda parameter}}
  __auto_type g = 2;
  struct BitField { int field:2; };
  __auto_type h = BitField{1}.field; // (should work from C++)
  new __auto_type; // expected-error {{'__auto_type' not allowed in type allocated by 'new'}}
}

