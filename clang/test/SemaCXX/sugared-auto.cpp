// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++20

enum class N {};

using Animal = int;

using AnimalPtr = Animal *;

using Man = Animal;
using Dog = Animal;

namespace variable {

auto x1 = Animal();
N t1 = x1; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

auto x2 = AnimalPtr();
N t2 = x2; // expected-error {{lvalue of type 'AnimalPtr' (aka 'int *')}}

auto *x3 = AnimalPtr();
N t3 = x3; // expected-error {{lvalue of type 'Animal *' (aka 'int *')}}

// Each variable deduces separately.
auto x4 = Man(), x5 = Dog();
N t4 = x4; // expected-error {{lvalue of type 'Man' (aka 'int')}}
N t5 = x5; // expected-error {{lvalue of type 'Dog' (aka 'int')}}

} // namespace variable

namespace function_basic {

auto f1() { return Animal(); }
auto x1 = f1();
N t1 = x1; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

decltype(auto) f2() { return Animal(); }
auto x2 = f2();
N t2 = x2; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

auto x3 = [a = Animal()] { return a; }();
N t3 = x3; // expected-error {{lvalue of type 'Animal' (aka 'int')}}

} // namespace function_basic
