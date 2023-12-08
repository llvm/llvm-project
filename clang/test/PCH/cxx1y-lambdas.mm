// RUN: %clang_cc1 -pedantic-errors -fblocks -std=c++1y -emit-pch %s -o %t-cxx1y
// RUN: %clang_cc1 -ast-print -pedantic-errors -fblocks -std=c++1y -include-pch %t-cxx1y  %s | FileCheck -check-prefix=CHECK-PRINT %s

#ifndef HEADER_INCLUDED

#define HEADER_INCLUDED
template<typename T>
T add_slowly(const T& x, const T &y) {
  return [](auto z, int y = 0) { return z + y; }(5);
};

inline int add_int_slowly_twice(int x, int y) {
  int i = add_slowly(x, y);
  auto lambda = [](auto z) { return z + z; };
  return i + lambda(y);
}

inline int sum_array(int n) {
  auto lambda = [](auto N) -> int {
    int sum = 0;
    int array[5] = { 1, 2, 3, 4, 5};
  
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

auto with_pack = [](auto ...xs){};

#else

// CHECK-PRINT: T add_slowly
// CHECK-PRINT: return []
template float add_slowly(const float&, const float&);

int add(int x, int y) {
  return add_int_slowly_twice(x, y) + sum_array(4) + to_block_pointer(5);
}

// CHECK-PRINT: inline int add_int_slowly_twice 
// CHECK-PRINT: lambda = [](auto z

// CHECK-PRINT: init_capture
// CHECK-PRINT: [&, x(t)]

void use_with_pack() { with_pack(1, 2, 3); }

#endif
