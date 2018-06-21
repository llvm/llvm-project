//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// See GCC PR63723.
// UNSUPPORTED: gcc-4.9

// <experimental/simd>
//
// [simd.class]
// template <class G> explicit simd(G&& gen);

#include <cstdint>
#include <experimental/simd>

using namespace std::experimental::parallelism_v2;

template <class T, class... Args>
auto not_supported_native_simd_ctor(Args&&... args)
    -> decltype(native_simd<T>(std::forward<Args>(args)...), void()) = delete;

template <class T>
void not_supported_native_simd_ctor(...) {}

template <class T, class... Args>
auto supported_native_simd_ctor(Args&&... args)
    -> decltype(native_simd<T>(std::forward<Args>(args)...), void()) {}

template <class T>
void supported_native_simd_ctor(...) = delete;

void compile_generator() {
  supported_native_simd_ctor<int>([](int i) { return i; });
  not_supported_native_simd_ctor<int>([](int i) { return float(i); });
  not_supported_native_simd_ctor<int>([](intptr_t i) { return (int*)(i); });
  not_supported_native_simd_ctor<int>([](int* i) { return i; });
}

int main() {}
