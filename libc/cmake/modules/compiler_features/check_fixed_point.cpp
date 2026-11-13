#include "include/llvm-libc-macros/stdfix-macros.h"

#ifndef LIBC_COMPILER_HAS_FIXED_POINT
#error unsupported
#endif

template <typename T> void check_fixed_point_mangling(T) {}
template void check_fixed_point_mangling(unsigned short fract);
