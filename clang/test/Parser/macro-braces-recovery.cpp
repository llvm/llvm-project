// RUN: %clang_cc1 -fsyntax-only -verify=expected -std=c++17 %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx20 -std=c++20 %s

namespace GH21755 {
#define M(x) f x // expected-note {{macro 'M' defined here}}

// expected-error@+5 {{too many arguments provided to function-like macro invocation}}
// expected-note@+4 {{parentheses are required around macro argument containing braced initializer list}}
// expected-error@+3 {{a type specifier is required for all declarations}}
// expected-error@+2 {{expected ')'}}
// expected-note@+1 {{to match this '('}}
M(0 {,}) // expected-error {{expected ';' after top level declarator}}
}

namespace GH115007 {
class Foo { // expected-note {{candidate constructor (the implicit copy constructor) not viable}} \
            // expected-note {{candidate constructor (the implicit move constructor) not viable}}
public:
  Foo(int); // expected-note {{candidate constructor not viable: requires 1 argument, but 2 were provided}}
  bool operator==(const int l); // expected-note {{candidate function not viable: no known conversion from 'Foo' to 'const int' for 1st argument}} \
                                // cxx20-note {{candidate function (with reversed parameter order) not viable: no known conversion from 'Foo' to 'const int' for object argument}}
};
#define EQ(x,y) (void)(x == y) // expected-note {{macro 'EQ' defined here}}

void test_EQ() {
  Foo F = Foo{1};
  // expected-error@+4 {{too many arguments provided to function-like macro invocation}}
  // expected-note@+3 {{parentheses are required around macro argument containing braced initializer list}}
  // expected-error@+2 {{no matching constructor for initialization of 'Foo'}}
  // expected-error@+1 {{invalid operands to binary expression ('Foo' and 'Foo')}}
  EQ(F,Foo{1,2});
}
}
