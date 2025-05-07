
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void unsafe_char(const char *__unsafe_indexable);

void unsafe_int(const int *__unsafe_indexable);

// expected-note@+1{{passing argument to parameter here}}
void nul_char(const char *__null_terminated);

// expected-note@+1{{passing argument to parameter here}}
void x_char(const char *__terminated_by('X'));

// expected-note@+1{{passing argument to parameter here}}
void nul_int(const int *__null_terminated);

// __null_terminated char pointer <-> __unsafe_indexable char pointer

char *__null_terminated unsafe_indexable_to_null_terminated_char(char *__unsafe_indexable p) {
  // expected-error@+1{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__unsafe_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  char *__null_terminated q = p;

  // expected-error@+1{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from incompatible type 'char *__unsafe_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  q = p;

  // expected-error@+1{{passing 'char *__unsafe_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  nul_char(p);

  // expected-error@+1{{returning 'char *__unsafe_indexable' from a function with incompatible result type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  return p;

  // expected-error@+1{{casting 'char *__unsafe_indexable' to incompatible type 'char * __terminated_by(0)' (aka 'char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  (void)((char *__null_terminated)p);
}

char *__unsafe_indexable null_terminated_char_to_unsafe_indexable(char *__null_terminated p) {
  // ok
  char *__unsafe_indexable q = p;

  // ok
  q = p;

  // ok
  unsafe_char(p);

  // ok
  return p;

  // ok
  (void)((char *__unsafe_indexable)p);
}

// __terminated_by('X') char pointer <-> __unsafe_indexable char pointer

char *__terminated_by('X') unsafe_indexable_to_terminated_by_char(char *__unsafe_indexable p) {
  // expected-error@+1{{initializing 'char *__single __terminated_by('X')' (aka 'char *__single') with an expression of incompatible type 'char *__unsafe_indexable' is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  char *__terminated_by('X') q = p;

  // expected-error@+1{{assigning to 'char *__single __terminated_by('X')' (aka 'char *__single') from incompatible type 'char *__unsafe_indexable' is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  q = p;

  // expected-error@+1{{passing 'char *__unsafe_indexable' to parameter of incompatible type 'const char *__single __terminated_by('X')' (aka 'const char *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  x_char(p);

  // expected-error@+1{{returning 'char *__unsafe_indexable' from a function with incompatible result type 'char *__single __terminated_by('X')' (aka 'char *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  return p;

  // expected-error@+1{{casting 'char *__unsafe_indexable' to incompatible type 'char * __terminated_by('X')' (aka 'char *') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  (void)((char *__terminated_by('X'))p);
}

char *__unsafe_indexable terminated_by_char_to_unsafe_indexable(char *__terminated_by('X') p) {
  char *__unsafe_indexable q = p;

  q = p;

  unsafe_char(p);

  return p;

  (void)((char *__unsafe_indexable)p);
}

// __null_terminated int pointer <-> __unsafe_indexable int pointer

int *__null_terminated unsafe_indexable_to_null_terminated_int(int *__unsafe_indexable p) {
  // expected-error@+1{{initializing 'int *__single __terminated_by(0)' (aka 'int *__single') with an expression of incompatible type 'int *__unsafe_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  int *__null_terminated q = p;

  // expected-error@+1{{assigning to 'int *__single __terminated_by(0)' (aka 'int *__single') from incompatible type 'int *__unsafe_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  q = p;

  // expected-error@+1{{passing 'int *__unsafe_indexable' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  nul_int(p);

  // expected-error@+1{{returning 'int *__unsafe_indexable' from a function with incompatible result type 'int *__single __terminated_by(0)' (aka 'int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  return p;

  // expected-error@+1{{casting 'int *__unsafe_indexable' to incompatible type 'int * __terminated_by(0)' (aka 'int *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  (void)((int *__null_terminated)p);
}

int *__unsafe_indexable null_terminated_int_to_unsafe_indexable(int *__null_terminated p) {
  int *__unsafe_indexable q = p;

  q = p;

  unsafe_int(p);

  return p;

  (void)((int *__unsafe_indexable)p);
}
