// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s -Wno-uninitialized

struct A {};

struct BASE {
  operator A(); // expected-note {{candidate function}}
};

struct BASE1 {
 operator A();  // expected-note {{candidate function}}
};

class B : public BASE , public BASE1
{
  public:
  B();
} b;

extern B f();

const int& ri = (void)0; // expected-error {{reference to type 'const int' could not bind to an rvalue of type 'void'}}

int main() {
        const A& rca = f(); // expected-error {{reference initialization of type 'const A &' with initializer of type 'B' is ambiguous}}
        A& ra = f(); // expected-error {{non-const lvalue reference to type 'A' cannot bind to a temporary of type 'B'}}
}

struct PR6139 { A (&x)[1]; };
PR6139 x = {{A()}}; // expected-error{{non-const lvalue reference to type 'A[1]' cannot bind to an initializer list temporary}}

struct PR6139b { A (&x)[1]; };
PR6139b y = {A()}; // expected-error{{non-const lvalue reference to type 'A[1]' cannot bind to a temporary of type 'A'}}

namespace PR16502 {
  struct A { int &&temporary; int x, y; };
  int f();
  const A &c = { 10, ++c.temporary };
}

namespace IncompleteTest {
  struct String;
  // expected-error@+1 {{reference to incomplete type 'const String' could not bind to an lvalue of type 'const char[1]'}}
  void takeString(const String& = "") {} // expected-note {{passing argument to parameter here}}
  void test() {
        takeString();
  }
}
