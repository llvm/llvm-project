//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/simd>
//
// [simd.casts]
// template <class T, class U, class Abi> see below static_simd_cast(const simd<U, Abi>&);

#include <experimental/simd>
#include <cstdint>

using namespace std::experimental::parallelism_v2;

static_assert(
    std::is_same<decltype(static_simd_cast<float>(native_simd<int>())),
                 native_simd<float>>::value,
    "");

static_assert(std::is_same<decltype(static_simd_cast<fixed_size_simd<float, 1>>(
                               simd<int, simd_abi::scalar>())),
                           fixed_size_simd<float, 1>>::value,
              "");

static_assert(
    std::is_same<decltype(static_simd_cast<simd<float, simd_abi::scalar>>(
                     fixed_size_simd<int, 1>())),
                 simd<float, simd_abi::scalar>>::value,
    "");

int main() {}
