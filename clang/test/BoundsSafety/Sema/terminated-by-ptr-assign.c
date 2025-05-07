
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

// expected-note@+1 +{{passing argument to parameter here}}
void nul(const char *__null_terminated);

void nul_c(const char *const __null_terminated);

// expected-note@+1 +{{passing argument to parameter here}}
void nul_int(const int *__null_terminated);

// expected-note@+1{{passing argument to parameter here}}
void _42(const char *__terminated_by(42));

// expected-note@+1{{passing argument to parameter here}}
void _42_int(const int *__terminated_by(42));

// expected-note@+1 +{{passing argument to parameter here}}
void nul_nested(char *__null_terminated *__null_terminated);

void _42_nested(char *__terminated_by(42) * __null_terminated);

// Null

char *__null_terminated null(void) {
  char *__null_terminated p = 0;      // ok
  p = 0;                              // ok
  nul(0);                             // ok
  return 0;                           // ok
  (void)((char *__null_terminated)0); // ok
}

// Char arrays
const char *__null_terminated const_arr_stringlit(int x) {
  const char const_arr_stringlit[] = "hello";

  const char *__null_terminated p = const_arr_stringlit;       // ok
  p = const_arr_stringlit;                                     // ok
  nul(const_arr_stringlit);                                    // ok
  (void)((const char *__null_terminated) const_arr_stringlit); // ok
  return const_arr_stringlit;                                  // ok
}

const char *__null_terminated const_arr_stringlit_arit(int x) {
  const char const_arr_stringlit[] = "hello";

  const char *__null_terminated p = const_arr_stringlit + 3;         // ok
  p = const_arr_stringlit + 3;                                       // ok
  nul(const_arr_stringlit + 3);                                      // ok
  (void)((const char *__null_terminated)( const_arr_stringlit + 3)); // ok
  return const_arr_stringlit + 3;                                    // ok
}

const char *__null_terminated const_arr_stringlit_arit_end(int x) {
  const char const_arr_stringlit[] = "hello";

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = const_arr_stringlit + 6;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = const_arr_stringlit + 6;
  // expected-error@+3{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(const_arr_stringlit + 6);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated)( const_arr_stringlit + 6));
  // expected-error@+3{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return const_arr_stringlit + 6;
}

const char *__null_terminated const_arr_stringlit_arit_pastend(int x) {
  const char const_arr_stringlit[] = "hello";

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = const_arr_stringlit + 7;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = const_arr_stringlit + 7;
  // expected-error@+3{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(const_arr_stringlit + 7);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated)( const_arr_stringlit + 7));
  // expected-error@+3{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return const_arr_stringlit + 7;
}

const char *__null_terminated const_arr_braced(int x) {
  const char const_arr_braced[] = "hello";

  const char *__null_terminated p = const_arr_braced;       // ok
  p = const_arr_braced;                                     // ok
  nul(const_arr_braced);                                    // ok
  (void)((const char *__null_terminated) const_arr_braced); // ok
  return const_arr_braced;                                  // ok
}

const char *__null_terminated const_arr_stringlit_nonnt_trunc(int x) {
  const char const_arr_stringlit_nonnt[2] = "he";

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char[2]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = const_arr_stringlit_nonnt;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char[2]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = const_arr_stringlit_nonnt;
  // expected-error@+3{{passing 'const char[2]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(const_arr_stringlit_nonnt);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) const_arr_stringlit_nonnt);
  // expected-error@+3{{returning 'const char[2]' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return const_arr_stringlit_nonnt;
}

const char *__null_terminated const_arr_braced_nonnt_trunc(int x) {
  const char const_arr_braced_nonnt[2] = {'h', 'e'};

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char[2]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = const_arr_braced_nonnt;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char[2]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = const_arr_braced_nonnt;
  // expected-error@+3{{passing 'const char[2]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(const_arr_braced_nonnt);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) const_arr_braced_nonnt);
  // expected-error@+3{{returning 'const char[2]' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return const_arr_braced_nonnt;
}

const char *__null_terminated const_arr_stringlit_nt(int x) {
  const char const_arr_stringlit_nt[3] = "he";

  const char *__null_terminated p = const_arr_stringlit_nt;
  p = const_arr_stringlit_nt;
  nul(const_arr_stringlit_nt);
  (void)((const char *__null_terminated) const_arr_stringlit_nt);
  return const_arr_stringlit_nt;
}

const char *__null_terminated const_arr_braced_implicit_nt(int x) {
  const char const_arr_braced_implicit_nt[3] = {'h', 'e'};

  const char *__null_terminated p = const_arr_braced_implicit_nt;
  p = const_arr_braced_implicit_nt;
  nul(const_arr_braced_implicit_nt);
  (void)((const char *__null_terminated) const_arr_braced_implicit_nt);
  return const_arr_braced_implicit_nt;
}

const char *__null_terminated const_arr_stringlit_nt_oversized(int x) {
  const char const_arr_stringlit_nt_oversized[4] = "he";

  const char *__null_terminated p = const_arr_stringlit_nt_oversized;
  p = const_arr_stringlit_nt_oversized;
  nul(const_arr_stringlit_nt_oversized);
  (void)((const char *__null_terminated) const_arr_stringlit_nt_oversized);
  return const_arr_stringlit_nt_oversized;
}

const char *__null_terminated const_arr_braced_nonnt_oversized(int x) {
  const char const_arr_braced_nonnt[4] = {'h', 'e', '\0', 'w'};

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char[4]' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = const_arr_braced_nonnt;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char[4]' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = const_arr_braced_nonnt;
  // expected-error@+3{{passing 'const char[4]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(const_arr_braced_nonnt);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) const_arr_braced_nonnt);
  // expected-error@+3{{returning 'const char[4]' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return const_arr_braced_nonnt;
}

const char *__null_terminated arr_stringlit(int x) {
  // expected-note@+1 4{{consider adding 'const' to 'arr_stringlit'}}
  char arr_stringlit[] = "hello";

  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'char[6]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = arr_stringlit;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'char[6]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = arr_stringlit;
  // expected-error@+3{{passing 'char[6]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(arr_stringlit);
  // expected-error@+3{{casting 'char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) arr_stringlit);
  // expected-error@+3{{returning 'char[6]' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return arr_stringlit;
}

// Compound literal

const int *__null_terminated compound_lit_explicit_ok(void) {
  const int *__null_terminated p = (const int[3]){1, 2, 0};       // ok
  p = (const int[3]){1, 2, 0};                                    // ok
  nul_int((const int[3]){1, 2, 0});                               // ok
  (void)((const int *__null_terminated) (const int[3]){1, 2, 0}); // ok
  return (const int[3]){1, 2, 0};                                 // ok
}

const int *__null_terminated compound_lit_explicit_bad(void) {
  // expected-error@+3{{initializing 'const int *__single __terminated_by(0)' (aka 'const int *__single') with an expression of incompatible type 'const int[3]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const int *__null_terminated p = (const int[3]){1, 2, 3};
  // expected-error@+3{{assigning to 'const int *__single __terminated_by(0)' (aka 'const int *__single') from incompatible type 'const int[3]' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = (const int[3]){1, 2, 3};
  // expected-error@+3{{passing 'const int[3]' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul_int((const int[3]){1, 2, 3});
  // expected-error@+3{{casting 'const int *__bidi_indexable' to incompatible type 'const int * __terminated_by(0)' (aka 'const int *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const int *__null_terminated) (const int[3]){1, 2, 3});
  // expected-error@+3{{returning 'const int[3]' from a function with incompatible result type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return (const int[3]){1, 2, 3};
}

const int *__null_terminated compound_lit_implicit_ok(void) {
  const int *__null_terminated p = (const int[3]){1, 2};       // ok
  p = (const int[3]){1, 2};                                    // ok
  nul_int((const int[3]){1, 2});                               // ok
  (void)((const int *__null_terminated) (const int[3]){1, 2}); // ok
  return (const int[3]){1, 2};                                 // ok
}

const int *__terminated_by(42) compound_lit_implicit_bad(void) {
  // expected-error@+1{{initializing 'const int *__single __terminated_by(42)' (aka 'const int *__single') with an expression of incompatible type 'const int[3]' is an unsafe operation}}
  const int *__terminated_by(42) p = (const int[3]){1, 2};
  // expected-error@+1{{assigning to 'const int *__single __terminated_by(42)' (aka 'const int *__single') from incompatible type 'const int[3]' is an unsafe operation}}
  p = (const int[3]){1, 2};
  // expected-error@+1{{passing 'const int[3]' to parameter of incompatible type 'const int *__single __terminated_by(42)' (aka 'const int *__single') is an unsafe operation}}
  _42_int((const int[3]){1, 2});
  // expected-error@+1{{casting 'const int *__bidi_indexable' to incompatible type 'const int * __terminated_by(42)' (aka 'const int *') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  (void)((const int *__terminated_by(42)) (const int[3]){1, 2});
  // expected-error@+1{{returning 'const int[3]' from a function with incompatible result type 'const int *__single __terminated_by(42)' (aka 'const int *__single') is an unsafe operation}}
  return (const int[3]){1, 2};
}

const int *__null_terminated compound_lit_arith_ok(void) {
  const int *__null_terminated p = (const int[3]){1, 2, 0} + 2;         // ok
  p = (const int[3]){1, 2, 0} + 2;                                      // ok
  nul_int((const int[3]){1, 2, 0} + 2);                                 // ok
  (void)((const int *__null_terminated) ((const int[3]){1, 2, 0} + 2)); // ok
  return (const int[3]){1, 2, 0} + 2;                                   // ok
}

const int *__null_terminated compound_lit_arith_end(void) {
  // expected-error@+3{{initializing 'const int *__single __terminated_by(0)' (aka 'const int *__single') with an expression of incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const int *__null_terminated p = (const int[3]){1, 2, 3} + 3;
  // expected-error@+3{{assigning to 'const int *__single __terminated_by(0)' (aka 'const int *__single') from incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = (const int[3]){1, 2, 3} + 3;
  // expected-error@+3{{passing 'const int *__bidi_indexable' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul_int((const int[3]){1, 2, 3} + 3);
  // expected-error@+3{{casting 'const int *__bidi_indexable' to incompatible type 'const int * __terminated_by(0)' (aka 'const int *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const int *__null_terminated) ((const int[3]){1, 2, 3} + 3));
  // expected-error@+3{{returning 'const int *__bidi_indexable' from a function with incompatible result type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return (const int[3]){1, 2, 3} + 3;
}

const int *__null_terminated compound_lit_arith_past_end(void) {
  // expected-error@+3{{initializing 'const int *__single __terminated_by(0)' (aka 'const int *__single') with an expression of incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const int *__null_terminated p = (const int[3]){1, 2, 3} + 4;
  // expected-error@+3{{assigning to 'const int *__single __terminated_by(0)' (aka 'const int *__single') from incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = (const int[3]){1, 2, 3} + 4;
  // expected-error@+3{{passing 'const int *__bidi_indexable' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul_int((const int[3]){1, 2, 3} + 4);
  // expected-error@+3{{casting 'const int *__bidi_indexable' to incompatible type 'const int * __terminated_by(0)' (aka 'const int *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const int *__null_terminated) ((const int[3]){1, 2, 3} + 4));
  // expected-error@+3{{returning 'const int *__bidi_indexable' from a function with incompatible result type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return (const int[3]){1, 2, 3} + 4;
}

const int *__null_terminated compound_lit_arith_neg(void) {
  // expected-error@+3{{initializing 'const int *__single __terminated_by(0)' (aka 'const int *__single') with an expression of incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const int *__null_terminated p = (const int[3]){1, 2, 3} - 1;
  // expected-error@+3{{assigning to 'const int *__single __terminated_by(0)' (aka 'const int *__single') from incompatible type 'const int *__bidi_indexable' is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = (const int[3]){1, 2, 3} - 1;
  // expected-error@+3{{passing 'const int *__bidi_indexable' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul_int((const int[3]){1, 2, 3} - 1);
  // expected-error@+3{{casting 'const int *__bidi_indexable' to incompatible type 'const int * __terminated_by(0)' (aka 'const int *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const int *__null_terminated) ((const int[3]){1, 2, 3} - 1));
  // expected-error@+3{{returning 'const int *__bidi_indexable' from a function with incompatible result type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return (const int[3]){1, 2, 3} - 1;
}

// String literal

char *__null_terminated string_literal_nul(void) {
  char *__null_terminated p = "init";                // ok
  p = "assign";                                      // ok
  nul("passing");                                    // ok
  return "returning";                                // ok
  (void)((const char *__null_terminated) "casting"); // ok
}

char *__terminated_by(42) string_literal_42(void) {
  char *__terminated_by(42) p = "init";                // expected-error{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  p = "assign";                                        // expected-error{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  _42("passing");                                      // expected-error{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  return "returning";                                  // expected-error{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  // expected-error@+1{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  (void)((const char *__terminated_by(42)) "casting");
}

// Terminators

char *__null_terminated terminators(char *__terminated_by(42) p, char *__null_terminated q) {
  // expected-error@+1{{pointers with incompatible terminators initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__single __terminated_by(42)' (aka 'char *__single')}}
  char *__null_terminated a = p;

  // ok
  char *__null_terminated b = q;

  // expected-error@+1{{pointers with incompatible terminators assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from incompatible type 'char *__single __terminated_by(42)' (aka 'char *__single')}}
  a = p;

  // ok
  a = q;

  // expected-error@+1{{pointers with incompatible terminators passing 'char *__single __terminated_by(42)' (aka 'char *__single') to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single')}}
  nul(p);

  // ok
  _42(p);

  // expected-error@+1{{pointers with incompatible terminators returning 'char *__single __terminated_by(42)' (aka 'char *__single') from a function with incompatible result type 'char *__single __terminated_by(0)' (aka 'char *__single')}}
  return p;

  // ok
  return q;

  // expected-error@+1{{pointers with incompatible terminators casting 'char *__single __terminated_by(42)' (aka 'char *__single') to incompatible type 'char * __terminated_by(0)' (aka 'char *')}}
  (void)((char *__null_terminated)p);

  // ok
  (void)((char *__null_terminated)q);
}

char *__null_terminated *__null_terminated nested_terminators(char *__terminated_by(42) * __null_terminated p, char *__null_terminated *__null_terminated q) {
  // expected-error@+1{{pointers with incompatible terminators initializing 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') with an expression of incompatible type 'char *__single __terminated_by(42)*__single __terminated_by(0)' (aka 'char *__single*__single')}}
  char *__null_terminated *__null_terminated a = p;

  // ok
  char *__null_terminated *__null_terminated b = q;

  // expected-error@+1{{pointers with incompatible terminators assigning to 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') from incompatible type 'char *__single __terminated_by(42)*__single __terminated_by(0)' (aka 'char *__single*__single')}}
  a = p;

  // ok
  a = q;

  // expected-error@+1{{pointers with incompatible terminators passing 'char *__single __terminated_by(42)*__single __terminated_by(0)' (aka 'char *__single*__single') to parameter of incompatible type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single')}}
  nul_nested(p);

  // ok
  _42_nested(p);

  // expected-error@+1{{pointers with incompatible terminators returning 'char *__single __terminated_by(42)*__single __terminated_by(0)' (aka 'char *__single*__single') from a function with incompatible result type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single')}}
  return p;

  // ok
  return q;

  // expected-error@+1{{pointers with incompatible terminators casting 'char *__single __terminated_by(42)*__single __terminated_by(0)' (aka 'char *__single*__single') to incompatible type 'char * __terminated_by(0)* __terminated_by(0)' (aka 'char **')}}
  (void)((char *__null_terminated *__null_terminated)p);

  // ok
  (void)((char *__null_terminated *__null_terminated)q);
}

// Quals

char *__null_terminated quals_c_to_nc(char *const __null_terminated p) {
  // ok
  char *__null_terminated a = p;

  // ok
  a = p;

  // ok
  nul(p);

  // ok
  (void)((char *__null_terminated)p);

  // ok
  return p;
}

char *const __null_terminated quals_nc_to_c(char *__null_terminated p) {
  // ok
  char *const __null_terminated a = p;

  // expected-note@-2{{variable 'a' declared const here}}
  // expected-error@+1{{cannot assign to variable 'a' with const-qualified type 'char *__single __terminated_by(0)const' (aka 'char *__singleconst')}}
  a = p;

  // ok
  nul_c(p);

  // ok
  (void)((char *const __null_terminated)p);

  // ok
  return p;
}

// Non-TerminatedBy to TerminatedBy

char *__null_terminated single_to_terminated_by(char *__single p) {
  // expected-error@+1{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  char *__null_terminated q = p;

  // expected-error@+1{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from incompatible type 'char *__single' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  q = p;

  // expected-error@+1{{passing 'char *__single' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  nul(p);

  // expected-error@+1{{returning 'char *__single' from a function with incompatible result type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  return p;

  // expected-error@+1{{casting 'char *__single' to incompatible type 'char * __terminated_by(0)' (aka 'char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  (void)((char *__null_terminated)p);
}

char *__null_terminated indexable_to_terminated_by(char *__indexable p) {
  // expected-error@+3{{initializing 'char *__single __terminated_by(0)' (aka 'char *__single') with an expression of incompatible type 'char *__indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  char *__null_terminated q = p;

  // expected-error@+3{{assigning to 'char *__single __terminated_by(0)' (aka 'char *__single') from incompatible type 'char *__indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  q = p;

  // expected-error@+3{{passing 'char *__indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(p);

  // expected-error@+3{{returning 'char *__indexable' from a function with incompatible result type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return p;

  // expected-error@+3{{casting 'char *__indexable' to incompatible type 'char * __terminated_by(0)' (aka 'char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((char *__null_terminated)p);
}

char *__null_terminated *__null_terminated nested_to_terminated_by(char *__single *__null_terminated p) {
  // expected-error@+1{{initializing 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') with an expression of incompatible type 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
  char *__null_terminated *__null_terminated q = p;

  // expected-error@+1{{assigning to 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') from incompatible type 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
  q = p;

  // expected-error@+1{{passing 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') to parameter of incompatible type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
  nul_nested(p);

  // expected-error@+1{{returning 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') from a function with incompatible result type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') that adds '__terminated_by' attribute is not allowed}}
  return p;

  // expected-error@+1{{casting 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') to incompatible type 'char * __terminated_by(0)* __terminated_by(0)' (aka 'char **') that adds '__terminated_by' attribute is not allowed}}
  (void)((char *__null_terminated *__null_terminated)p);
}

// TerminatedBy to Non-TerminatedBy

// expected-note@+1{{passing argument to parameter here}}
void foo_single(char *__single);

// expected-note@+1{{passing argument to parameter here}}
void foo_indexable(char *__indexable);

char *__single single_from_terminated_by(char *__null_terminated p) {
  // expected-error@+1{{initializing 'char *__single' with an expression of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  char *__single q = p;

  // expected-error@+1{{assigning to 'char *__single' from incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  q = p;

  // expected-error@+1{{passing 'char *__single __terminated_by(0)' (aka 'char *__single') to parameter of incompatible type 'char *__single' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  foo_single(p);

  // expected-error@+1{{returning 'char *__single __terminated_by(0)' (aka 'char *__single') from a function with incompatible result type 'char *__single' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  return p;

  // expected-error@+1{{casting 'char *__single __terminated_by(0)' (aka 'char *__single') to incompatible type 'char *__single' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  (void)((char *__single)p);
}

char *__indexable indexable_from_terminated_by(char *__null_terminated p) {
  // expected-error@+3{{initializing 'char *__indexable' with an expression of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  char *__indexable q = p;

  // expected-error@+3{{assigning to 'char *__indexable' from incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  q = p;

  // expected-error@+3{{passing 'char *__single __terminated_by(0)' (aka 'char *__single') to parameter of incompatible type 'char *__indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  foo_indexable(p);

  // expected-error@+3{{returning 'char *__single __terminated_by(0)' (aka 'char *__single') from a function with incompatible result type 'char *__indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  return p;

  // expected-error@+3{{casting 'char *__single __terminated_by(0)' (aka 'char *__single') to incompatible type 'char *__indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+2{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  (void)((char *__indexable)p);
}

// expected-note@+1{{passing argument to parameter here}}
void bar(char *__single *__null_terminated);

char *__single *__null_terminated nested_from_terminated_by(char *__null_terminated *__null_terminated p) {
  // expected-error@+1{{initializing 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') with an expression of incompatible type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') that discards '__terminated_by' attribute is not allowed}}
  char *__single *__null_terminated q = p;

  // expected-error@+1{{assigning to 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') from incompatible type 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') that discards '__terminated_by' attribute is not allowed}}
  q = p;

  // expected-error@+1{{passing 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') to parameter of incompatible type 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') that discards '__terminated_by' attribute is not allowed}}
  bar(p);

  // expected-error@+1{{returning 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') from a function with incompatible result type 'char *__single*__single __terminated_by(0)' (aka 'char *__single*__single') that discards '__terminated_by' attribute is not allowed}}
  return p;

  // expected-error@+1{{casting 'char *__single __terminated_by(0)*__single __terminated_by(0)' (aka 'char *__single*__single') to incompatible type 'char *__single* __terminated_by(0)' (aka 'char *__single*') that discards '__terminated_by' attribute is not allowed}}
  (void)((char *__single *__null_terminated)p);
}

void sign_mismatch(void) {
  const unsigned char array[] = "foo";
  // expected-warning@+1{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of type 'const unsigned char[4]' converts between pointers to integer types where one is of the unique plain 'char' type and the other is not}}
  const char *__terminated_by(0) p = array;
}

// Conditional operators

char *__null_terminated cond_op_string_lit(int cond) {
  char *__null_terminated p = cond ? "init-true" : "init-false";                     // ok
  p = cond ? "assign-true" : "assign-false";                                         // ok
  nul(cond ? "passing-true" : "passing-false");                                      // ok
  (void)((const char *__null_terminated) (cond ? "casting-true" : "casting-false")); // ok
  return cond ? "returning-true" : "returning-false";                                // ok
}

char *__null_terminated cond_op_nt_with_string_lit(int cond, char *__null_terminated s) {
  char *__null_terminated p = cond ? s : "init-false";                  // ok
  p = cond ? s : "assign-false";                                        // ok
  nul(cond ? s : "passing-false");                                      // ok
  (void)((const char *__null_terminated) (cond ? s : "casting-false")); // ok
  return cond ? s : "returning-false";                                  // ok
}

char *__null_terminated cond_op_string_lit_nested(int c, int d, int e) {
  char *__null_terminated p = c ? (d ? "tt" : "tf") : (e ? "ft" : "ff");              // ok
  p = c ? (d ? "tt" : "tf") : (e ? "ft" : "ff");                                      // ok
  nul(c ? (d ? "tt" : "tf") : (e ? "ft" : "ff"));                                     // ok
  (void)((const char *__null_terminated)(c ? (d ? "tt" : "tf") : (e ? "ft" : "ff"))); // ok
  return c ? (d ? "tt" : "tf") : (e ? "ft" : "ff");                                   // ok
}

const char *__null_terminated cond_op_char_array(int cond) {
  const char t[] = "true";
  const char f[] = "false";
  const char *__null_terminated p = cond ? t : f;         // ok
  p = cond ? t : f;                                       // ok
  nul(cond ? t : f);                                      // ok
  (void)((const char *__null_terminated) (cond ? t : f)); // ok
  return cond ? t : f;                                    // ok
}

const char *__null_terminated cond_op_char_array_arith(int cond) {
  const char t[] = "true";
  const char f[] = "false";
  const char *__null_terminated p = cond ? t+3 : f+4;         // ok
  p = cond ? t+3 : f+4;                                       // ok
  nul(cond ? t+3 : f+4);                                      // ok
  (void)((const char *__null_terminated) (cond ? t+3 : f+4)); // ok
  return cond ? t+3 : f+4;                                    // ok
}

const char *__null_terminated cond_op_char_array_arith_true_end(int cond) {
  const char t[] = "true";
  const char f[] = "false";
  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = cond ? t+5 : f+4;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = cond ? t+5 : f+4;
  // expected-error@+3{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(cond ? t+5 : f+4);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) (cond ? t+5 : f+4));
  // expected-error@+3{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return cond ? t+5 : f+4;
}

const char *__null_terminated cond_op_char_array_arith_false_end(int cond) {
  const char t[] = "true";
  const char f[] = "false";
  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = cond ? t+4 : f+6;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = cond ? t+4 : f+6;
  // expected-error@+3{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(cond ? t+4 : f+6);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) (cond ? t+4 : f+6));
  // expected-error@+3{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return cond ? t+4 : f+6;
}

const char *__null_terminated cond_op_char_array_wrong_term(int cond) {
  const char t[] = "true";
  const char f[5] = "false"; // not NUL-terminated
  // expected-error@+3{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  const char *__null_terminated p = cond ? t : f;
  // expected-error@+3{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  p = cond ? t : f;
  // expected-error@+3{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  nul(cond ? t : f);
  // expected-error@+3{{casting 'const char *__bidi_indexable' to incompatible type 'const char * __terminated_by(0)' (aka 'const char *') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  (void)((const char *__null_terminated) (cond ? t : f));
  // expected-error@+3{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  return cond ? t : f;
}

const int *__null_terminated cond_op_different_type(int cond) {
  const int i[] = {1, 2, 3, 0};
  const long l[] = {1, 2, 0};
  // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'const int *__bidi_indexable' and 'const long *__bidi_indexable'; use explicit casts to perform this conversion}}
  const long *__null_terminated p = cond ? i : l;
  // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'const int *__bidi_indexable' and 'const long *__bidi_indexable'; use explicit casts to perform this conversion}}
  p = cond ? i : l;
  // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'const int *__bidi_indexable' and 'const long *__bidi_indexable'; use explicit casts to perform this conversion}}
  nul_int(cond ? i : l);
  // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'const int *__bidi_indexable' and 'const long *__bidi_indexable'; use explicit casts to perform this conversion}}
  (void)((const int *__null_terminated) (cond ? i : l));
  // expected-error@+1{{conditional expression evaluates values with incompatible pointee types 'const int *__bidi_indexable' and 'const long *__bidi_indexable'; use explicit casts to perform this conversion}}
  return cond ? i : l;
}

char *__null_terminated gnu_cond_op_string_lit_true(void) {
  char *__null_terminated p = "t" ?: "f";               // ok
  p = "t" ?: "f";                                       // ok
  nul("t" ?: "f");                                      // ok
  (void)((const char *__null_terminated) ("t" ?: "f")); // ok
  return "t" ?: "f";                                    // ok
}

char *__null_terminated gnu_cond_op_string_lit_false(void) {
  char *__null_terminated p = 0 ?: "f";               // ok
  p = 0 ?: "f";                                       // ok
  nul(0 ?: "f");                                      // ok
  (void)((const char *__null_terminated) (0 ?: "f")); // ok
  return 0 ?: "f";                                    // ok
}

void const_ptr_indexable_to_terminated_by(int x) {
  const char *__indexable ptr_terminated_by;;

  // expected-error@+1{{initializing 'const char *__single __terminated_by(4)' (aka 'const char *__single') with an expression of incompatible type 'const char *__indexable' is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  const char *__terminated_by(4) p = ptr_terminated_by;
}
