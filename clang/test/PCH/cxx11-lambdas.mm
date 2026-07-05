// RUN: %clang_cc1 -pedantic-errors -fblocks -std=c++1y -emit-pch %s -o %t-cxx11
// RUN: %clang_cc1 -ast-print -pedantic-errors -fblocks -std=c++1y -include-pch %t-cxx11  %s | FileCheck -check-prefix=CHECK-PRINT %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
template<typename T>
T add_slowly(const T& x, const T &y) {
  return [=, &y] { return x + y; }();
};

inline int add_int_slowly_twice(int x, int y) {
  int i = add_slowly(x, y);
  auto lambda = [&](int z) { return x + z; };
  return i + lambda(y);
}

inline int sum_array(int n) {
  int array[5] = { 1, 2, 3, 4, 5};
  auto lambda = [=](int N) -> int {
    int sum = 0;
    for (unsigned I = 0; I < N; ++I)
      sum += array[N];
    return sum;
  };

  return lambda(n);
}

inline int to_block_pointer(int n) {
  auto lambda = [=](int m) { return n + m; };
  int (^block)(int) = lambda;
  return block(17);
}

template<typename T>
int init_capture(T t) {
  return [&, x(t)] { return sizeof(x); };
}

struct X {
  template <typename T> X(T) {}
};
struct Y { Y(const X &x = [] {}); };

#else

// CHECK-PRINT: T add_slowly
// CHECK-PRINT: return [=, &y]
template float add_slowly(const float&, const float&);

int add(int x, int y) {
  return add_int_slowly_twice(x, y) + sum_array(4) + to_block_pointer(5);
}

// CHECK-PRINT: inline int add_int_slowly_twice 
// CHECK-PRINT: lambda = [&](int z)

// CHECK-PRINT: init_capture
// CHECK-PRINT: [&, x(t)]

X x = [] {};

#endif
