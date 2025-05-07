

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

// expected-no-diagnostics

#include <ptrcheck.h>

// pass_dynamic_object_size attribute only applies to constant pointer
// arguments, make sure that after handling __null_terminated the pointer
// remains const.

unsigned long my_strlen_c(const char *const s __attribute__((pass_dynamic_object_size(0)))) {
  return __builtin_strlen(s);
}

unsigned long my_strlen_c_nt(const char *const __null_terminated s __attribute__((pass_dynamic_object_size(0)))) {
  return __builtin_strlen(s);
}

unsigned long my_strlen_nt_c(const char *__null_terminated const s __attribute__((pass_dynamic_object_size(0)))) {
  return __builtin_strlen(s);
}
