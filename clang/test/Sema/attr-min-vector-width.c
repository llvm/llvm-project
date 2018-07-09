// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

int i;
void f(void) __attribute__((__min_vector_width__(i))); /* expected-error {{'__min_vector_width__' attribute requires an integer constant}} */

void f2(void) __attribute__((__min_vector_width__(128)));

void f3(void) __attribute__((__min_vector_width__(128), __min_vector_width__(256))); /* expected-warning {{attribute '__min_vector_width__' is already applied with different parameters}} */

void f4(void) __attribute__((__min_vector_width__())); /* expected-error {{'__min_vector_width__' attribute takes one argument}} */

void f5(void) __attribute__((__min_vector_width__(128, 256))); /* expected-error {{'__min_vector_width__' attribute takes one argument}} */

void f6(void) {
  int x __attribute__((__min_vector_width__(128))) = 0; /* expected-error {{'__min_vector_width__' attribute only applies to functions}} */
}
