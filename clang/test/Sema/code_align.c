// RUN: %clang_cc1 -fsyntax-only -verify=expected,c-local -x c %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp-local -pedantic -x c++ -std=c++11 %s

void foo() {
  int i;
  int a[10], b[10];

  [[clang::code_align(8)]]
  for (i = 0; i < 10; ++i) {  // this is OK
    a[i] = b[i] = 0;
  }
  // expected-error@+1{{'code_align' attribute only applies to 'for', 'while', and 'do' statements}}
  [[clang::code_align(4)]]
  i = 7;
  for (i = 0; i < 10; ++i) {
    a[i] = b[i] = 0;
  }

  // expected-error@+1{{'code_align' attribute cannot be applied to a declaration}}
  [[clang::code_align(12)]] int n[10];
}

void bar(int);
// cpp-local-note@+1{{declared here}}
void foo1(int A)
{
  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; got 0}}
  [[clang::code_align(0)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; got -4}}
  [[clang::code_align(-4)]]
  for(int I=0; I<128; ++I) { bar(I); }

    // cpp-local-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
    // c-local-error@+1{{integer constant expression must have integer type, not 'double'}}
  [[clang::code_align(64.0)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute takes one argument}}
  [[clang::code_align()]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute takes one argument}}
  [[clang::code_align(4,8)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // no diagnostic is expected
  [[clang::code_align(32)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+2{{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}
  // c-local-error@+1{{integer constant expression must have integer type, not 'char[4]'}}
  [[clang::code_align("abc")]]
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(64)]] // expected-note{{previous attribute is here}}
  [[clang::code_align(64)]] // expected-error{{duplicate loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; got 7}}
  [[clang::code_align(7)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; got 5000}}
  [[clang::code_align(5000)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+3{{expression is not an integral constant expression}}
  // cpp-local-note@+2{{function parameter 'A' with unknown value cannot be used in a constant expression}}
  // c-local-error@+1{{expression is not an integer constant expression}}
  [[clang::code_align(A)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

void check_code_align_expression() {
  int a[10];

  // Test that checks expression is not a constant expression.
  int foo2; // cpp-local-note {{declared here}}
  // c-local-error@+3{{expression is not an integer constant expression}}
  // cpp-local-error@+2{{expression is not an integral constant expression}}
  // cpp-local-note@+1{{read of non-const variable 'foo2' is not allowed in a constant expression}}
  [[clang::code_align(foo2 + 1)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

#if __cplusplus >= 201103L
  // Test that checks expression is a constant expression.
  constexpr int bars = 0;
  [[clang::code_align(bars + 1)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
#endif
}

#if __cplusplus >= 201103L
template <int A, int B, int C, int D>
void code_align_dependent() {
  [[clang::code_align(C)]]
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]] // expected-note{{previous attribute is here}}
  [[clang::code_align(B)]] // cpp-local-error{{duplicate loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+2{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; got -10}}
  // cpp-local-note@#neg-instantiation{{in instantiation of function template specialization}}
  [[clang::code_align(D)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

int main() {
  code_align_dependent<8, 16, 32, -10>(); // #neg-instantiation
  return 0;
}
#endif
