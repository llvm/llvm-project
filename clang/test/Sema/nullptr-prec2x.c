// RUN: %clang_cc1 -fsyntax-only -verify -std=c17 -ffreestanding -Wc2x-compat %s

#include <stddef.h>

int nullptr; // expected-warning {{'nullptr' is a keyword in C2x}}

nullptr_t val; // expected-error {{unknown type name 'nullptr_t'}}

