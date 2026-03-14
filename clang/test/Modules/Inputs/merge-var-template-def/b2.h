#ifndef B2_H
#define B2_H

template<typename T, T v>
struct S { static constexpr T value = v; };
template<typename T, T v>
constexpr T S<T, v>::value;

#endif
