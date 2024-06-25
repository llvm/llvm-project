// RUN: %clang_cc1 -triple x86_64-pc-linux -fsyntax-only -verify=expected,c-local -x c %s
// RUN: %clang_cc1 -triple x86_64-pc-linux -fsyntax-only -verify=expected,cpp-local -pedantic -x c++ -std=c++11 %s

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
  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was 0}}
  [[clang::code_align(0)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was -4}}
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

  [[clang::code_align(64)]] // OK
  [[clang::code_align(64)]] // OK
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(8)]]  // expected-note{{previous attribute is here}}
  [[clang::code_align(64)]] // expected-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(4)]] // expected-note{{previous attribute is here}}
  [[clang::code_align(4)]] // OK
  [[clang::code_align(8)]] // expected-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(4)]]  // expected-note 2{{previous attribute is here}}
  [[clang::code_align(4)]]  // OK
  [[clang::code_align(8)]]  // expected-error{{conflicting loop attribute 'code_align'}}
  [[clang::code_align(64)]] // expected-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was 7}}
  [[clang::code_align(7)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was 5000}}
  [[clang::code_align(5000)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // expected-warning@+2{{integer literal is too large to be represented in a signed integer type, interpreting as unsigned}}
  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was -9223372036854775808}}
  [[clang::code_align(9223372036854775808)]]
  for(int I=0; I<256; ++I) { bar(I); }

#ifdef __SIZEOF_INT128__
  // expected-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was (__int128_t)1311768467294899680ULL << 64}}
  [[clang::code_align((__int128_t)0x1234567890abcde0ULL << 64)]]
  for(int I=0; I<256; ++I) { bar(I); }
#endif

  // expected-error@+1 {{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was -922337203685477}}
  [[clang::code_align(-922337203685477)]]
  for(int I=0; I<256; ++I) { bar(I); }

#ifdef __SIZEOF_INT128__
  // cpp-local-error@+3{{expression is not an integral constant expression}}
  // cpp-local-note@+2{{left shift of negative value -1311768467294899680}}
  // c-local-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was -(__int128_t)1311768467294899680ULL << 64}}
  [[clang::code_align(-(__int128_t)0x1234567890abcde0ULL << 64)]]
  for(int I=0; I<256; ++I) { bar(I); }
#endif

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
template <int A, int B, int C, int D, int E>
void code_align_dependent() {
  [[clang::code_align(C)]]
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]] // OK
  [[clang::code_align(A)]] // OK
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]] // cpp-local-note{{previous attribute is here}}
  [[clang::code_align(E)]] // cpp-local-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]] // cpp-local-note{{previous attribute is here}}
  [[clang::code_align(A)]] // OK
  [[clang::code_align(E)]] // cpp-local-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  [[clang::code_align(A)]] // cpp-local-note 2{{previous attribute is here}}
  [[clang::code_align(A)]] // OK
  [[clang::code_align(C)]] // cpp-local-error{{conflicting loop attribute 'code_align'}}
  [[clang::code_align(E)]] // cpp-local-error{{conflicting loop attribute 'code_align'}}
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+1{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was 23}}
  [[clang::code_align(B)]]
  for(int I=0; I<128; ++I) { bar(I); }

  // cpp-local-error@+2{{'code_align' attribute requires an integer argument which is a constant power of two between 1 and 4096 inclusive; provided argument was -10}}
  // cpp-local-note@#neg-instantiation{{in instantiation of function template specialization 'code_align_dependent<8, 23, 32, -10, 64>' requested here}}
  [[clang::code_align(D)]]
  for(int I=0; I<128; ++I) { bar(I); }
}

template<int ITMPL>
void bar3() {
  [[clang::code_align(8)]]      // cpp-local-note{{previous attribute is here}}
  [[clang::code_align(ITMPL)]] // cpp-local-error{{conflicting loop attribute 'code_align'}} \
	                       // cpp-local-note@#temp-instantiation{{in instantiation of function template specialization 'bar3<4>' requested here}}
  for(int I=0; I<128; ++I) { bar(I); }
}

template<int ITMPL1>
void bar4() {
  [[clang::code_align(ITMPL1)]] // cpp-local-note{{previous attribute is here}}
  [[clang::code_align(32)]]    // cpp-local-error{{conflicting loop attribute 'code_align'}} \
	                       // cpp-local-note@#temp-instantiation1{{in instantiation of function template specialization 'bar4<64>' requested here}}
  for(int I=0; I<128; ++I) { bar(I); }
}

int main() {
  code_align_dependent<8, 23, 32, -10, 64>(); // #neg-instantiation
  bar3<4>();  // #temp-instantiation
  bar4<64>(); // #temp-instantiation1
  return 0;
}
#endif
