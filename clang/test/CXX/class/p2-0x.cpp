// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
namespace Test1 {

class A final { }; // expected-note {{'A' declared here}}
class B : A { }; // expected-error {{base 'A' is marked 'final'}}

}

namespace Test2 {

template<typename T> struct A final { }; // expected-note 2 {{'A' declared here}}
struct B : A<int> { }; // expected-error {{base 'A' is marked 'final'}}
  
template<typename T> struct C : A<T> { }; // expected-error {{base 'A' is marked 'final'}}
struct D : C<int> { }; // expected-note {{in instantiation of template class 'Test2::C<int>' requested here}}

}

namespace Test3 {

template<typename T> struct A { };
template<> struct A<int> final { }; // expected-note {{'A' declared here}}

struct B : A<bool> { };
struct C : A<int> { }; // expected-error {{base 'A' is marked 'final'}}

}

namespace Test4 {

struct A final { virtual void func() = 0; }; // expected-warning {{abstract class is marked 'final'}} expected-note {{unimplemented pure virtual method 'func' in 'A'}}
struct B { virtual void func() = 0; }; // expected-note {{unimplemented pure virtual method 'func' in 'C'}}

struct C final : B { }; // expected-warning {{abstract class is marked 'final'}}

}
