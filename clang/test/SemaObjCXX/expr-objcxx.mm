// RUN: %clang_cc1 %s -verify -pedantic -fsyntax-only
// expected-no-diagnostics

// rdar://8366474
void *P =  @selector(foo::bar::);
