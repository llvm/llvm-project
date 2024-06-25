// RUN: %clang_cc1 -verify -pedantic -std=c99 %s

void func(void) {
  _Complex float cf;
  _Complex double cd;
  _Complex long double cld;

  ++cf;  // expected-warning {{'++' on an object of complex type is a Clang extension}}
  ++cd;  // expected-warning {{'++' on an object of complex type is a Clang extension}}
  ++cld; // expected-warning {{'++' on an object of complex type is a Clang extension}}

  --cf;  // expected-warning {{'--' on an object of complex type is a Clang extension}}
  --cd;  // expected-warning {{'--' on an object of complex type is a Clang extension}}
  --cld; // expected-warning {{'--' on an object of complex type is a Clang extension}}

  cf++;  // expected-warning {{'++' on an object of complex type is a Clang extension}}
  cd++;  // expected-warning {{'++' on an object of complex type is a Clang extension}}
  cld++; // expected-warning {{'++' on an object of complex type is a Clang extension}}

  cf--;  // expected-warning {{'--' on an object of complex type is a Clang extension}}
  cd--;  // expected-warning {{'--' on an object of complex type is a Clang extension}}
  cld--; // expected-warning {{'--' on an object of complex type is a Clang extension}}
}

