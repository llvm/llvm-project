// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

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
PR6139 x = {{A()}}; // expected-error{{non-const lvalue reference to type 'A [1]' cannot bind to a temporary of type 'A'}}
