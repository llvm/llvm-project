

// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// __null_terminated should be inherited.
void inherit(const char *__null_terminated p) {
  const char *__null_terminated q1 = (const char *)p;
  const char *__null_terminated q2 = (char *)p;
}

// __null_terminated shouldn't be inherited.
void dont_inherit(const char *__null_terminated p) {
  // expected-error@+1{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const int *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  const char *__null_terminated q1 = (const int *)p;

  // expected-error@+1{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'int *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  const char *__null_terminated q2 = (int *)p;

  // expected-error@+1{{pointers with incompatible terminators casting 'const char *__single __terminated_by(0)' (aka 'const char *__single') to incompatible type 'const char * __terminated_by(42)' (aka 'const char *')}}
  const char *__null_terminated q3 = (const char *__terminated_by(42))p;

  // expected-error@+1{{casting 'const char *__single __terminated_by(0)' (aka 'const char *__single') to incompatible type 'const char *__single' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  const char *__null_terminated q4 = (const char *__single)p;

  // expected-error@+3{{casting 'const char *__single __terminated_by(0)' (aka 'const char *__single') to incompatible type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  const char *__null_terminated q5 = (const char *__bidi_indexable)p;
}

void inherit_nested(const char *__null_terminated *__null_terminated p) {
  const char *__null_terminated *__null_terminated q1 = (const char **)p;
  const char *__null_terminated *__null_terminated q2 = (const char *__null_terminated *)p;
  const char *__null_terminated *__null_terminated q3 = (const char **__null_terminated)p;
}

void dont_inherit_nested(const char *__null_terminated *__null_terminated p) {
  // expected-error@+1{{initializing 'const char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'const char *__single*__single') with an expression of incompatible type 'const int *__single*__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  const char *__null_terminated *__null_terminated q1 = (const int **)p;

  // expected-error@+1{{pointers with incompatible terminators casting 'const char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'const char *__single*__single') to incompatible type 'const char * __terminated_by(42)*' (aka 'const char **')}}
  const char *__null_terminated *__null_terminated q2 = (const char *__terminated_by(42) *)p;

  // expected-error@+1{{casting 'const char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'const char *__single*__single') to incompatible type 'const char *__single*' that discards '__terminated_by' attribute is not allowed}}
  const char *__null_terminated *__null_terminated q3 = (const char *__single *)p;
}

void test_terminated_by(const char *__terminated_by(8) p) {
  // expected-error@+1{{casting 'const char *__single __terminated_by(8)' (aka 'const char *__single') to incompatible type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__terminated_by_to_indexable()' to perform this conversion explicitly}}
  const char *__bidi_indexable q = (const char *__bidi_indexable)p;
}
