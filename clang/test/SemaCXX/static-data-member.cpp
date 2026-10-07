// RUN: %clang_cc1 %std_cxx98- -fsyntax-only -verify -w %s

struct ABC {
  static double a;
  static double b;
  static double c;
  static double d;
  static double e;
  static double f;
};

double ABC::a = 1.0;
extern double ABC::b = 1.0; // expected-error {{static data member definition cannot specify a storage class}}
static double ABC::c = 1.0;  // expected-error {{'static' can only be specified inside the class definition}}
__private_extern__ double ABC::d = 1.0; // expected-error {{static data member definition cannot specify a storage class}}
auto double ABC::e = 1.0; // expected-error {{static data member definition cannot specify a storage class}}
#if __cplusplus < 201703L
register double ABC::f = 1.0; // expected-error {{static data member definition cannot specify a storage class}}
#endif
