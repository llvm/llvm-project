/* FIXME: This is a file containing various typos for which we can
   suggest corrections but are unable to actually recover from
   them. Ideally, we would eliminate all such cases and move these
   tests elsewhere. */

// RUN: %clang_cc1 -fsyntax-only -verify %s

float f(int y) {
  return static_cst<float>(y); // expected-error{{use of undeclared identifier 'static_cst'; did you mean 'static_cast'?}} \
  // expected-error{{for function-style cast or type construction}}
}

