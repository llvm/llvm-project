// RUN: %clang_cc1 -fsyntax-only -verify %s

void f() {
  @throw @"Hello"; // expected-error {{cannot use '@throw' with Objective-C exceptions disabled}}
}

void g() {
  @try { // expected-error {{cannot use '@try' with Objective-C exceptions disabled}}
    f();
  } @finally {
    
  }
}
