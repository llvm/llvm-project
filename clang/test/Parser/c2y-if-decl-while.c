// RUN: %clang_cc1 -std=c2y -pedantic -Wpre-c2y-compat -verify %s
// RUN: %clang_cc1 -std=c23 -pedantic -verify=expected,ext %s
// RUN: %clang_cc1 -std=c17 -pedantic -verify=expected,ext %s

// N3356 ('if' declarations) only applies to selection statements. A declaration
// in the controlling clause of a 'while' (or 'do'/'while') statement must be
// rejected rather than accepted as a C2y extension.

void test(void) {
  if (int x = 1) {} // ext-warning {{'if' declaration statements are a C2y extension}}
  switch (int x = 1) {} // ext-warning {{'switch' declaration statements are a C2y extension}}

  while (int x = 12) {} // expected-error {{expected expression}}
  while (_Static_assert(1, ""); 1) {} // expected-error {{expected expression}}
  do {} while (int x = 1); // expected-error {{expected expression}}
}
