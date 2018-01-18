// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 \
// RUN: -triple powerpc64le-unknown-linux-gnu -target-cpu pwr8 \
// RUN: -target-feature +float128 %s
// RUN: %clang_cc1 -fsyntax-only -std=c++11 -triple x86_64-unknown-linux-gnu -Wno-unused-value -Wno-parentheses %s

__float128 qf();
long double ldf();

#ifdef __PPC__
// FIXME: once operations between long double and __float128 are implemented for
//        targets where the types are different, these next two will change
long double ld{qf()}; // expected-error {{cannot initialize a variable of type 'long double' with an rvalue of type '__float128'}}
__float128 q{ldf()};  // expected-error {{cannot initialize a variable of type '__float128' with an rvalue of type 'long double'}}

auto test1(__float128 q, long double ld) -> decltype(q + ld) { // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
  return q + ld;      // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
}

auto test2(long double a, __float128 b) -> decltype(a + b) { // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
  return a + b;      // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
}
#endif

void test3(bool b) {
  long double ld;
  __float128 q;

  ld + q; // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
  q + ld; // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
  ld - q; // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
  q - ld; // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
  ld * q; // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
  q * ld; // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
  ld / q; // expected-error {{invalid operands to binary expression ('long double' and '__float128')}}
  q / ld; // expected-error {{invalid operands to binary expression ('__float128' and 'long double')}}
  ld = q; // expected-error {{assigning to 'long double' from incompatible type '__float128'}}
  q = ld; // expected-error {{assigning to '__float128' from incompatible type 'long double'}}
  q + b ? q : ld; // expected-error {{incompatible operand types ('__float128' and 'long double')}}
}
