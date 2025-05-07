
#include <va-list-sys.h>

// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include
// RUN: %clang_cc1 -fbounds-safety %s -verify -I %S/include -x objective-c -fexperimental-bounds-safety-objc
// expected-no-diagnostics
extern variable_length_function func_ptr;
typedef void * (*variable_length_function2)(va_list args);
extern variable_length_function2 func_ptr2;

void func(char *dst_str, char *src_str, int len) {
  call_func(func_ptr, dst_str, src_str, len);
  call_func(func_ptr2, dst_str, src_str, len);
}
