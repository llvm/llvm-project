// RUN: %clang_cc1 -fsyntax-only -verify=expected,c-local %s
// RUN: %clang_cc1 -fsyntax-only -verify=expected,cpp-local -pedantic -x c++ -std=c++11 %s 

// Add diagnostics tests for Loop attribute: [[clang::code_align()]].

void foo() {
  int i;
  int a[10], b[10];

  [[clang::code_align(8)]]
  for (i = 0; i < 10; ++i) {  // this is OK
    a[i] = b[i] = 0;
  }
  // expected-error@+1 {{'code_align' attribute only applies to 'for', 'while', and 'do' statements}}
  [[clang::code_align(4)]]
  i = 7;
  for (i = 0; i < 10; ++i) {
    a[i] = b[i] = 0;
  }

  // expected-error@+1{{'code_align' attribute cannot be applied to a declaration}}
  [[clang::code_align(12)]] int n[10];
}

void bar(int);
#if __cplusplus >= 201103L
// cpp-local-note@+2 {{declared here}}
#endif
void foo1(int A)
{
  // expected-error@+1 {{'code_align' attribute requires a positive integral compile time constant expression}}
  [[clang::code_align(0)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires a positive integral compile time constant expression}}
  [[clang::code_align(-4)]]
  for(int I=0; I<128; ++I) { bar(I); }

#if __cplusplus >= 201103L
    // cpp-local-error@+4 {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
#else
    // c-local-error@+2 {{integer constant expression must have integer type, not 'double'}}
#endif
  [[clang::code_align(64.0)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1 {{'code_align' attribute takes one argument}}
  [[clang::code_align()]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1 {{'code_align' attribute takes one argument}}
  [[clang::code_align(4,8)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // no diagnostic is expected
  [[clang::code_align(32)]]
  for(int I=0; I<128; ++I) { bar(I); }

#if __cplusplus >= 201103L
  // cpp-local-error@+4 {{integral constant expression must have integral or unscoped enumeration type, not 'const char[4]'}}
#else  
  // c-local-error@+2 {{integer constant expression must have integer type, not 'char[4]'}}
#endif
  [[clang::code_align("abc")]]
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(64)]]
  // expected-error@+1 {{duplicate loop attribute 'code_align'}}
  [[clang::code_align(64)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1 {{'code_align' attribute argument must be a constant power of two greater than zero}}
  [[clang::code_align(7)]]
  for(int I=0; I<128; ++I) { bar(I); }

#if __cplusplus >= 201103L
  // cpp-local-error@+5 {{expression is not an integral constant expression}}
  // cpp-local-note@+4 {{function parameter 'A' with unknown value cannot be used in a constant expression}}
#else
  // c-local-error@+2 {{expression is not an integer constant expression}}
#endif  
  [[clang::code_align(A)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

#if __cplusplus >= 201103L
void check_code_align_expression() {
  int a[10];

  // Test that checks expression is not a constant expression.
  int foo2; // cpp-local-note {{declared here}}
  // cpp-local-error@+2{{expression is not an integral constant expression}}
  // cpp-local-note@+1{{read of non-const variable 'foo2' is not allowed in a constant expression}}
  [[clang::code_align(foo2 + 1)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;

  // Test that checks expression is a constant expression.
  constexpr int bars = 0;
  [[clang::code_align(bars + 1)]]
  for (int i = 0; i != 10; ++i)
    a[i] = 0;
}

template <int A, int B, int C, int D>
void code_align_dependent() {
  [[clang::code_align(C)]]
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]]
  // cpp-local-error@+1 {{duplicate loop attribute 'code_align'}}
  [[clang::code_align(B)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+1{{'code_align' attribute requires a positive integral compile time constant expression}}
  [[clang::code_align(D)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

int main() {
  code_align_dependent<8, 16, 32, -10>(); // cpp-local-note{{in instantiation of function template specialization 'code_align_dependent<8, 16, 32, -10>' requested here}}
  return 0;
}
#endif
