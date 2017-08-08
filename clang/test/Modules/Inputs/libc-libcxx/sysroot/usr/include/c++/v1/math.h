#ifndef LIBCXX_MATH_H
#define LIBCXX_MATH_H

#include_next <math.h>
template<typename T> T abs(T t) { return (t < 0) ? -t : t; }

#include <type_traits>

#endif
