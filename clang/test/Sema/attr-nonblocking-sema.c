// RUN: %clang_cc1 -fsyntax-only -verify -std=c89 %s

// Tests for a few cases involving C functions without prototypes.

void noproto() __attribute__((clang_nonblocking)) // expected-error {{'clang_nonblocking' function must have a prototype}}
{
}

// This will succeed
void noproto(void) __attribute__((clang_blocking));

// A redeclaration isn't any different - a prototype is required.
void f1(void);
void f1() __attribute__((clang_nonblocking)); // expected-error {{'clang_nonblocking' function must have a prototype}}
