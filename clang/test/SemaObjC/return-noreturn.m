// RUN: %clang_cc1 %s -fsyntax-only -fobjc-exceptions -verify -Wreturn-type -Wmissing-noreturn

id f(id self) {
} // expected-warning {{non-void function does not return a value}}

id f2(id self) {
  @try {
    @throw (id)0;
  } @catch (id) {
  }
  return (id)0;
}

