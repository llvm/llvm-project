// RUN: %clang_cc1 -std=c++20 %s -triple %itanium_abi_triple -emit-llvm -o - | FileCheck %s

module;
# 5 __FILE__ 1
class Pooh;
class Piglet;
# 8 "" 2

# 8 "" 1 3
export module std; // might happen, you can't say it won't!
# 9 "" 2 3

namespace std {
export template<typename T> class allocator {
// just for testing, not real!
void M (T *);
template <typename U> U *N (T *);
};

template<typename T> void allocator<T>::M (T *) {}
template<typename T> template<typename U> U *allocator<T>::N (T *) {
return nullptr;
}

// CHECK-DAG: void @_ZNStW3std9allocatorIiE1MEPi(
template void allocator<int>::M (int *);
// CHECK-DAG: @_ZNStW3std9allocatorIiE1NIfEEPT_Pi(
template float *allocator<int>::N<float> (int *);
}

// CHECK-DAG: @_ZNStW3std9allocatorI4PoohE1MEPS1_(
template void std::allocator<Pooh>::M (Pooh *);
// CHECK-DAG: @_ZNStW3std9allocatorI4PoohE1NI6PigletEEPT_PS1_(
template Piglet *std::allocator<Pooh>::N<Piglet> (Pooh *);
