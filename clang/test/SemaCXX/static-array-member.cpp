// RUN: %clang_cc1 -fsyntax-only %s

struct X0 {
  static int array[];
  
  int x;
  int y;
};

int X0::array[sizeof(X0) * 2];

template<typename T, int N>
struct X1 {
  static T array[];
};

template<typename T, int N>
T X1<T, N>::array[N];
