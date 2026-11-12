// RUN: %clang_cc1 -std=c++20 -verify -Wno-unused %s

template<typename T, typename U>
void operator->*(T, U);

template<typename T, typename U>
void operator+(T, U);

template<typename T, typename U>
void operator-(T, U);

template<typename T, typename U>
void operator*(T, U);

template<typename T, typename U>
void operator/(T, U);

template<typename T, typename U>
void operator%(T, U);

template<typename T, typename U>
void operator^(T, U);

template<typename T, typename U>
void operator&(T, U);

template<typename T, typename U>
void operator|(T, U);

template<typename T, typename U>
void operator+=(T, U);

template<typename T, typename U>
void operator-=(T, U);

template<typename T, typename U>
void operator*=(T, U);

template<typename T, typename U>
void operator/=(T, U);

template<typename T, typename U>
void operator%=(T, U);

template<typename T, typename U>
void operator^=(T, U);

template<typename T, typename U>
void operator&=(T, U);

template<typename T, typename U>
void operator|=(T, U);

template<typename T, typename U>
void operator==(T, U);

template<typename T, typename U>
void operator!=(T, U);

template<typename T, typename U>
void operator<(T, U);

template<typename T, typename U>
void operator>(T, U);

template<typename T, typename U>
void operator<=(T, U);

template<typename T, typename U>
void operator>=(T, U);

template<typename T, typename U>
void operator<=>(T, U);

template<typename T, typename U>
void operator&&(T, U);

template<typename T, typename U>
void operator||(T, U);

template<typename T, typename U>
void operator<<(T, U);

template<typename T, typename U>
void operator>>(T, U);

template<typename T, typename U>
void operator<<=(T, U);

template<typename T, typename U>
void operator>>=(T, U);

template<typename T, typename U>
void operator,(T, U);

template<typename T>
void operator*(T);

template<typename T>
void operator&(T);

template<typename T>
void operator+(T);

template<typename T>
void operator-(T);

template<typename T>
void operator!(T);

template<typename T>
void operator~(T);

template<typename T>
void operator++(T);

template<typename T>
void operator--(T);

template<typename T>
void operator++(T, int);

template<typename T>
void operator--(T, int);

template<typename T>
void f(int *x) {
  [&](auto *y) {
    *y;
    &y;
    +y;
    -y; // expected-error {{invalid argument type 'auto *' to unary expression}}
    !y;
    ~y; // expected-error {{invalid argument type 'auto *' to unary expression}}
    ++y;
    --y;
    y++;
    y--;
    y->*x;
    y + x;
    y - x;
    y * x;
    y / x;
    y % x;
    y ^ x;
    y & x;
    y | x;
    y += x;
    y -= x;
    y *= x;
    y /= x;
    y %= x;
    y ^= x;
    y &= x;
    y |= x;
    y == x;
    y != x;
    y < x;
    y > x;
    y <= x;
    y >= x;
    y <=> x;
    y && x;
    y || x;
    y << x;
    y >> x;
    y <<= x;
    y >>= x;
    y, x;
  };
}

template void f<int>(int*);
