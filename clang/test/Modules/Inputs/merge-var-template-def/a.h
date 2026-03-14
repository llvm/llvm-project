#ifndef A_H
#define A_H
template<typename T, T v>
struct S { static constexpr T value = v; };
template<typename T, T v>
constexpr T S<T, v>::value;

#endif
