// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only %s -fcxx-exceptions -fassume-nothrow-exception-dtor -verify

namespace test1 {
template <typename T> struct A { A(); ~A(); };
struct B { ~B() noexcept(false); };
struct B1 : B {};
struct B2 { B b; };
struct C { virtual void f(); } c;
struct MoveOnly { MoveOnly(); MoveOnly(MoveOnly&&); };
void run() {
  throw A<int>();
  throw B();  // expected-error{{cannot throw object of type 'B' with a potentially-throwing destructor}}
  throw new B;
  throw B1(); // expected-error{{cannot throw object of type 'B1' with a potentially-throwing destructor}}
  B2 b2;
  throw b2;   // expected-error{{cannot throw object of type 'B2' with a potentially-throwing destructor}}
  throw c;
  MoveOnly m;
  throw m;
}
}
