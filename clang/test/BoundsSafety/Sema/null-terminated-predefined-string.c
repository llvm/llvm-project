
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

void fnull_terminated(const char *);
void fnull_terminated_explicit(const char *__null_terminated);
// expected-note@+3{{passing argument to parameter here}}
// expected-note@+2{{passing argument to parameter here}}
// expected-note@+1{{passing argument to parameter here}}
void fnull_terminated_iptr(const int *__null_terminated);
// expected-note@+3{{passing argument to parameter here}}
// expected-note@+2{{passing argument to parameter here}}
// expected-note@+1{{passing argument to parameter here}}
void fterminated_by_minus_one(const int *__terminated_by(-1));

void test(void) {
  fnull_terminated(__func__);
  fnull_terminated(__FUNCTION__);
  fnull_terminated(__PRETTY_FUNCTION__);

  fnull_terminated_explicit(__func__);
  fnull_terminated_explicit(__FUNCTION__);
  fnull_terminated_explicit(__PRETTY_FUNCTION__);

  // expected-error@+3{{passing 'const char[5]' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  fnull_terminated_iptr(__func__);
  // expected-error@+3{{passing 'const char[5]' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  fnull_terminated_iptr(__FUNCTION__);
  // expected-error@+3{{passing 'const char[16]' to parameter of incompatible type 'const int *__single __terminated_by(0)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+1{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  fnull_terminated_iptr(__PRETTY_FUNCTION__);

  // expected-error@+1{{passing 'const char[5]' to parameter of incompatible type 'const int *__single __terminated_by(-1)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  fterminated_by_minus_one(__func__);
  // expected-error@+1{{passing 'const char[5]' to parameter of incompatible type 'const int *__single __terminated_by(-1)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  fterminated_by_minus_one(__FUNCTION__);
  // expected-error@+1{{passing 'const char[16]' to parameter of incompatible type 'const int *__single __terminated_by(-1)' (aka 'const int *__single') is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or '__unsafe_forge_terminated_by()' to perform this conversion}}
  fterminated_by_minus_one(__PRETTY_FUNCTION__);
}

void test_assign(const char *fname) {
  fname = __func__;
  fname = __FUNCTION__;
  fname = __PRETTY_FUNCTION__;

  const char *__null_terminated lfname1 = __func__;
  const char *__null_terminated lfname2 = __FUNCTION__;
  const char *__null_terminated lfname3 = __PRETTY_FUNCTION__;

  const char *lfname_non_null_terminated1 = __func__;
  const char *lfname_non_null_terminated2 = __FUNCTION__;
  const char *lfname_non_null_terminated3 = __PRETTY_FUNCTION__;

  // expected-error@+1{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  const char *__terminated_by(-1) lfname_term_minus_one1 = __func__;
  // expected-error@+1{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  const char *__terminated_by(-1) lfname_term_minus_one2 = __FUNCTION__;
  // expected-error@+1{{'__terminated_by' pointer converted from a string literal must be NUL-terminated}}
  const char *__terminated_by(-1) lfname_term_minus_one3 = __PRETTY_FUNCTION__;

}
