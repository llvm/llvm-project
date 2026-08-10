// RUN: %clang_cc1 -std=c++11 -verify %s

struct NotAggregateBase {};

struct A : NotAggregateBase {
private:
  A() = default; // expected-note {{here}}
};
A a = {}; // expected-error {{calling a private constructor}}

struct B : NotAggregateBase {
  explicit B() = default; // expected-note {{here}}
};
B b = {}; // expected-error {{chosen constructor is explicit}}
B b2{};
B b3;
