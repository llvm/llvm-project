// RUN: %clang_cc1 %s -verify -fsyntax-only


struct S {
[[clang::debug_transparent]]
void correct(void) {}

[[clang::debug_transparent(1)]] // expected-error {{'debug_transparent' attribute takes no arguments}}
void one_arg(void) {}
};

