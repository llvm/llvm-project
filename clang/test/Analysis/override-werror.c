// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -Werror %s -verify
// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -Werror %s -analyzer-werror -verify=werror

// This test case illustrates that using '-analyze' overrides the effect of
// -Werror.  This allows basic warnings not to interfere with producing
// analyzer results.

void f(int *p) {
  int; // expected-warning{{declaration does not declare anything}} \
          werror-warning{{declaration does not declare anything}}
}

void g(int *p) {
  if (!p) *p = 0; // expected-warning{{null}} \
                     werror-error{{null}}
}

