// Test this without pch.
// RUN: %clang_cc1 -include %S/Inputs/chain-macro-override1.h -include %S/Inputs/chain-macro-override2.h -fsyntax-only -verify -detailed-preprocessing-record %s

// Test with pch.
// RUN: %clang_cc1 -emit-pch -o %t1 %S/Inputs/chain-macro-override1.h -detailed-preprocessing-record
// RUN: %clang_cc1 -emit-pch -o %t2 %S/Inputs/chain-macro-override2.h -include-pch %t1 -detailed-preprocessing-record
// RUN: %clang_cc1 -include-pch %t2 -fsyntax-only -verify %s

int foo(void) {
  f();
  g();
  h();
  h2(); // expected-error {{call to undeclared function 'h2'; ISO C99 and later do not support implicit function declarations}}
  h3();
  return x;
}
