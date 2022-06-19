// RUN: %clang_cc1 -fsyntax-only -verify %s

void f(void) {
  @throw @"Hello"; // expected-error {{cannot use '@throw' with Objective-C exceptions disabled}}
}

void g(void) {
  @try { // expected-error {{cannot use '@try' with Objective-C exceptions disabled}}
    f();
  } @finally {
    
  }
}
