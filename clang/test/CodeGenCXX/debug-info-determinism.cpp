// RUN: %clang_cc1 -S -emit-llvm -g -o %t1.ll %s
// RUN: %clang_cc1 -S -emit-llvm -g -o %t2.ll %s
// RUN: diff %t1.ll %t2.ll

template <int N> struct C {
  template <int M> int f() {
    int arr[M] = {};
    return arr[M/2] + C<M/2>().template f<M-1>();
  }
};
template <> template <> int C<0>::f<0>() { return 0; }
template <> template <> int C<0>::f<1>() { return 0; }
template <> template <> int C<1>::f<0>() { return 0; }
template <> template <> int C<1>::f<1>() { return 0; }

int x = C<0>().f<64>();
