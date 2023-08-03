// RUN: %clang_cc1 %s -verify -fsyntax-only


struct S {
[[clang::transparent_stepping]]
void correct(void) {}

[[clang::transparent_stepping(1)]] // expected-error {{'transparent_stepping' attribute takes no arguments}}
void one_arg(void) {}
};

