#include "include/llvm-libc-macros/float16-macros.h"
#include "include/llvm-libc-types/float128.h"

#ifndef LIBC_TYPES_HAS_FLOAT16
#error unsupported
#endif

_Float16 cvt_from_float(float x) { return static_cast<_Float16>(x); }

_Float16 cvt_from_double(double x) { return static_cast<_Float16>(x); }

_Float16 cvt_from_long_double(long double x) {
  return static_cast<_Float16>(x);
}

#ifdef LIBC_TYPES_HAS_FLOAT128
_Float16 cvt_from_float128(float128 x) { return static_cast<_Float16>(x); }
#endif

float cvt_to_float(_Float16 x) { return x; }

double cvt_to_double(_Float16 x) { return x; }

long double cvt_to_long_double(_Float16 x) { return x; }

#ifdef LIBC_TYPES_HAS_FLOAT128
float128 cvt_to_float128(_Float16 x) { return x; }
#endif

extern "C" void _start() {}
