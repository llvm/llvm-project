// RUN: %clang_cc1 %s -verify -fsyntax-only

__attribute__((transparent_stepping))
void correct(void) {}

__attribute__((transparent_stepping(1))) // expected-error {{'transparent_stepping' attribute takes no arguments}}
void wrong_arg(void) {}
