// RUN: %clang_cc1 -std=c2y -pedantic -Wpre-c2y-compat -verify=expected,compat %s
// RUN: %clang_cc1 -std=c23 -pedantic -verify=expected,ext %s
// RUN: %clang_cc1 -std=c17 -pedantic -verify=expected,ext %s
// The extension warning is on by default, not only under -pedantic.
// RUN: %clang_cc1 -std=c23 -verify=expected,ext %s
// RUN: %clang_cc1 -std=c17 -verify=expected,ext %s
// RUN: %clang_cc1 -std=c17 -Wno-c2y-extensions -verify %s

// N3356 ('if' declarations) only applies to selection statements. A declaration
// in the controlling clause of a 'while' (or 'do'/'while') statement must be
// rejected rather than accepted as a C2y extension.

void test(void) {
  if (int x = 1) {} // ext-warning {{'if' declaration statements are a C2y extension}} \
                    // compat-warning {{'if' declaration statements are incompatible with C standards before C2y}}
  switch (int x = 1) {} // ext-warning {{'switch' declaration statements are a C2y extension}} \
                        // compat-warning {{'switch' declaration statements are incompatible with C standards before C2y}}

  while (int x = 12) {} // expected-error {{ISO C does not allow a declaration in a 'while' statement}}
  while (_Static_assert(1, ""); 1) {} // expected-error {{ISO C does not allow a declaration in a 'while' statement}}
  do {} while (int x = 1); // expected-error {{expected expression}}
}
