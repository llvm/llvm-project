// RUN: %clang_cc1 %s -verify -fsyntax-only

class C {
  void *f(int, int)
       __attribute__((ownership_returns(foo, 2)))
       __attribute__((ownership_returns(foo, 3))); // expected-error {{'ownership_returns' attribute arguments do not match the previous declaration}}
};
