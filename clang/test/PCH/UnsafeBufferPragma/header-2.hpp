#ifndef UNSAFE_HEADER
#define UNSAFE_HEADER

int foo(int *p) {
  return p[5];  // This will be warned
}

_Pragma("clang unsafe_buffer_usage begin") // The opt-out region spans over two files of one TU
#include "header-2.hpp"

#else

int bar(int *p) {
  return p[5];
}
_Pragma("clang unsafe_buffer_usage end")
#endif


