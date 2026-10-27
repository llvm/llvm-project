// RUN: %clang_cc1 -fsyntax-only -verify %s

struct S { S(int); operator bool(); };

void f() {
  int a;
  while (a) ;
  while (int x) ; // expected-error {{variable declaration in condition must have an initializer}}
  while (float x = 0) ;
  if (const int x = a) ; // expected-warning{{empty body}} expected-note{{put the semicolon on a separate line to silence this warning}}
  switch (int x = a+10) {}
  for (; int x = ++a; ) ;

  if (S a(42)) {} // expected-error {{variable declaration in condition cannot have a parenthesized initializer}}
}

#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wgnu-statement-expression"
#pragma clang diagnostic warning "-Wgnu-conditional-omitted-operand"

void extension_in_condition(int x, int y) {
  if (__extension__({ 0; })) {}
  if (__extension__(x ?: y) ==
      (x ?: y)) {} // expected-warning {{use of GNU ?: conditional expression extension, omitting middle operand}}
  if (__extension__ int z = (x ?: y);
      z == (x ?: y)) {} // expected-warning {{use of GNU ?: conditional expression extension, omitting middle operand}}
  for (; __extension__(x ?: y);) {}
}

#pragma clang diagnostic pop
