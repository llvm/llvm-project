// RUN: %clang_cc1 %s -verify -fsyntax-only

__attribute__((debug_transparent))
void correct(void) {}

__attribute__((debug_transparent(1))) // expected-error {{'debug_transparent' attribute takes no arguments}}
void wrong_arg(void) {}
