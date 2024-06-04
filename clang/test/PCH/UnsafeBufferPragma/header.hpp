#ifndef UNSAFE_HEADER
#define UNSAFE_HEADER
//This is a PCH file:

int foo(int *p) {
  return p[5];  // This will be warned
}

_Pragma("clang unsafe_buffer_usage begin")
#include "header.hpp"
_Pragma("clang unsafe_buffer_usage end")

#else
// This part is included by the PCH in the traditional way.  The
// include directive in the PCH is enclosed in an opt-out region, so
// unsafe operations here is suppressed.

int bar(int *p) {
  return p[5];
}

#endif


