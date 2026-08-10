// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -std=c++11 -verify %s
// RUN: %clang_cc1 -std=c++1y -verify %s

void div() {
  (void)(42 / 0); // expected-warning{{division by zero is undefined}}
  (void)(42 / false); // expected-warning{{division by zero is undefined}}
  (void)(42 / !1); // expected-warning{{division by zero is undefined}}
  (void)(42 / (1 - 1)); // expected-warning{{division by zero is undefined}}
  (void)(42 / !(1 + 1)); // expected-warning{{division by zero is undefined}}
  (void)(42 / (int)(0.0)); // expected-warning{{division by zero is undefined}}
}

void rem() {
  (void)(42 % 0); // expected-warning{{remainder by zero is undefined}}
  (void)(42 % false); // expected-warning{{remainder by zero is undefined}}
  (void)(42 % !1); // expected-warning{{remainder by zero is undefined}}
  (void)(42 % (1 - 1)); // expected-warning{{remainder by zero is undefined}}
  (void)(42 % !(1 + 1)); // expected-warning{{remainder by zero is undefined}}
  (void)(42 % (int)(0.0)); // expected-warning{{remainder by zero is undefined}}
}
