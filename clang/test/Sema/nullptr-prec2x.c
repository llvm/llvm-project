// RUN: %clang_cc1 -fsyntax-only -verify -std=c17 -ffreestanding -Wc2x-compat %s

#include <stddef.h>

int nullptr; // expected-warning {{'nullptr' is a keyword in C23}}

nullptr_t val; // expected-error {{unknown type name 'nullptr_t'}}

void foo(void *);
void bar() { foo(__nullptr); } // Test that it converts properly to an arbitrary pointer type without warning
_Static_assert(__nullptr == 0, "value of __nullptr"); // Test that its value matches that of NULL
_Static_assert(_Generic(__typeof(__nullptr), int : 0, void * : 0, default : 1), "type of __nullptr"); // Test that it's type is not the same as what NULL would generally have.
