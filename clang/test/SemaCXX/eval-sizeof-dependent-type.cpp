// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -std=c++11 -x c++ %s

typedef __SIZE_TYPE__ size_t;
template <typename _Tp, size_t _Nm> struct array { _Tp _M_elems[_Nm]; };
template <typename T> struct s {
  array<int, 1> v{static_cast<int>(sizeof (T) / sizeof(T))};
};

