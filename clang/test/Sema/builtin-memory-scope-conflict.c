// RUN: %clang_cc1 -fsyntax-only -std=c11 -verify %s

// Test that declaring __memory_scope conflicts with the builtin enum

// User declares their own enum first
typedef enum {
  my_value = 0
} __memory_scope; // expected-note {{'__memory_scope' declared here}}

// Trying to use builtin identifier will find the user's enum
// but the builtin enumerators won't be available
void test(void) {
  __memory_scope x = my_value;
  __memory_scope y = __memory_scope_system; // expected-warning {{user declaration of '__memory_scope' conflicts with builtin memory scope enum}} expected-error {{use of undeclared identifier '__memory_scope_system'}}
}
