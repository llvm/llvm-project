// RUN: %clang_cc1 %s -fsyntax-only -verify -std=c++11

struct X {};
typedef X foo_t;

foo_t *ptr;
char c1 = ptr; // expected-error{{'foo_t *' (aka 'X *')}}

const foo_t &ref = foo_t();
char c2 = ref; // expected-error{{'const foo_t' (aka 'const X')}}

// deduced auto should not produce an aka.
auto aut = X();
char c3 = aut; // expected-error{{from 'X' to 'char'}}

// There are two classes named Foo::foo here.  Make sure the message gives
// a way to them apart.
namespace Foo {
  class foo {};
}

namespace bar {
  namespace Foo {
    class foo;
  }
  void f(Foo::foo* x);  // expected-note{{passing argument to parameter 'x' here}}
}

void test(Foo::foo* x) {
  bar::f(x); // expected-error{{cannot initialize a parameter of type 'Foo::foo *' (aka 'bar::Foo::foo *') with an lvalue of type 'Foo::foo *'}}
}

namespace ns {
 struct str {
   static void method(struct data *) {}
 };
}

struct data { int i; };

typedef void (*callback)(struct data *);

void helper(callback cb) {} // expected-note{{candidate function not viable: no known conversion from 'void (*)(struct data *)' (aka 'void (*)(ns::data *)') to 'callback' (aka 'void (*)(struct data *)') for 1st argument}}

void test() {
 helper(&ns::str::method); // expected-error{{no matching function for call to 'helper'}}
}
