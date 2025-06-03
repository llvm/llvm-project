// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic
void check(int);
void a(void) {
  check(__builtin_isfinite(1.0f));
  check(__builtin_isinf(1.0));
  check(__builtin_isinf_sign(1.0L));
  check(__builtin_isnan(1.0f));
  check(__builtin_isnormal(1.0f));
  check(__builtin_isfinite(1)); // expected-error{{requires argument of floating point type}}
  check(__builtin_isinf()); // expected-error{{too few arguments}}
  check(__builtin_isnan(1,2)); // expected-error{{too many arguments}}
  check(__builtin_fpclassify(0, 0, 0, 0, 0, 1.0));
  check(__builtin_fpclassify(0, 0, 0, 0, 0, 1)); // expected-error{{requires argument of floating point type}}
  check(__builtin_fpclassify(0, 1, 2, 3, 4.5, 5.0)); // expected-warning{{implicit conversion from 'double' to 'int' changes value from 4.5 to 4}}
  check(__builtin_fpclassify(0, 0, 0, 0, 1)); // expected-error{{too few arguments}}
  check(__builtin_fpclassify(0, 0, 0, 0, 0, 1, 0)); // expected-error{{too many arguments}}

  check(__builtin_fpclassify(0,0,0,0,0, (invalid))); // expected-error{{use of undeclared identifier 'invalid'}}
  check(__builtin_fpclassify(0,0,0,0,0, (inf))); // expected-error{{use of undeclared identifier 'inf'}}
                                                // expected-error@-1{{reference to overloaded function could not be resolved}}
}
