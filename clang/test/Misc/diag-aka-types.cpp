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

template <typename T>
class A {};

int a1 = A<decltype(1 + 2)>(); // expected-error{{no viable conversion from 'A<decltype(1 + 2)>' (aka 'A<int>') to 'int'}}
int a2 = A<A<decltype(1 + 2)>>(); // expected-error{{no viable conversion from 'A<A<decltype(1 + 2)>>' (aka 'A<A<int>>') to 'int'}}
int a3 = A<__typeof(1 + 2)>(); // expected-error{{no viable conversion from 'A<typeof (1 + 2)>' (aka 'A<int>') to 'int'}}
int a4 = A<A<__typeof(1 + 2)>>(); // expected-error{{no viable conversion from 'A<A<typeof (1 + 2)>>' (aka 'A<A<int>>') to 'int'}}

using B = A<decltype(1+2)>;
int a5 = B(); // expected-error{{no viable conversion from 'B' (aka 'A<int>') to 'int'}}

decltype(void()) (&f1)() = 0; // expected-error{{non-const lvalue reference to type 'decltype(void()) ()' (aka 'void ()') cannot bind to a temporary of type 'int'}}
decltype(void()) (&f2)(int) = 0; // expected-error{{non-const lvalue reference to type 'decltype(void()) (int)' (aka 'void (int)') cannot bind to a temporary of type 'int'}}
void (&f3)(decltype(1 + 2)) = 0; // expected-error{{non-const lvalue reference to type 'void (decltype(1 + 2))' (aka 'void (int)') cannot bind to a temporary of type 'int'}}
decltype(1+2) (&f4)(double, decltype(1 + 2)) = 0; // expected-error{{non-const lvalue reference to type 'decltype(1 + 2) (double, decltype(1 + 2))' (aka 'int (double, int)') cannot bind to a temporary of type 'int'}}
auto (&f5)() -> decltype(1+2) = 0; // expected-error{{non-const lvalue reference to type 'auto () -> decltype(1 + 2)' (aka 'auto () -> int') cannot bind to a temporary of type 'int'}}

using C = decltype(1+2);;
C a6[10];
extern C a8[];
int a9 = a6; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'C[10]' (aka 'int[10]')}}
int a10 = a8; // expected-error{{cannot initialize a variable of type 'int' with an lvalue of type 'C[]' (aka 'int[]')}}
