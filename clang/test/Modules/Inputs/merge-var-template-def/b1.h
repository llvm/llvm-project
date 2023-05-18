#ifndef B1_H
#define B1_H
template<typename T, T v>
struct S { static constexpr T value = v; };
template<typename T, T v>
constexpr T S<T, v>::value;

#include "a.h"
#endif
