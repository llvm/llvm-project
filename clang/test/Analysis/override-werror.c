// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core -Werror %s -analyzer-store=region -verify

// This test case illustrates that using '-analyze' overrides the effect of
// -Werror.  This allows basic warnings not to interfere with producing
// analyzer results.

char* f(int *p) { 
  return p; // expected-warning{{incompatible pointer types}}
}

void g(int *p) {
  if (!p) *p = 0; // expected-warning{{null}}  
}

