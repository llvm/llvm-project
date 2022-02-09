// RUN: %clang_cc1 -Wdouble-promotion -Wimplicit-float-conversion %s -triple x86_64-apple-macosx10.12 -verify=x86,expected
// RUN: %clang_cc1 -Wdouble-promotion -Wimplicit-float-conversion %s -triple armv7-apple-ios9.0 -verify=arm,expected

// On ARM, long double and double both map to double precision 754s, so there
// isn't any reason to warn on conversions back and forth.

long double ld;
double d;
_Float16 f16; // x86-error {{_Float16 is not supported on this target}}

int main(void) {
  ld = d; // x86-warning {{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d = ld; // x86-warning {{implicit conversion loses floating-point precision: 'long double' to 'double'}}

  ld += d; // x86-warning {{implicit conversion increases floating-point precision: 'double' to 'long double'}}
  d += ld; // x86-warning {{implicit conversion when assigning computation result loses floating-point precision: 'long double' to 'double'}}

  f16 = ld; // expected-warning {{implicit conversion loses floating-point precision: 'long double' to '_Float16'}}
  ld = f16; // expected-warning {{implicit conversion increases floating-point precision: '_Float16' to 'long double'}}

  f16 += ld; // expected-warning {{implicit conversion when assigning computation result loses floating-point precision: 'long double' to '_Float16'}}
  ld += f16; // expected-warning {{implicit conversion increases floating-point precision: '_Float16' to 'long double'}}
}

