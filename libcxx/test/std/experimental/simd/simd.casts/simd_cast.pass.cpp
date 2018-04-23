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
// template <class T, class U, class Abi> see below simd_cast(const simd<U, Abi>&);
#include <experimental/simd>
#include <cstdint>

using namespace std::experimental::parallelism_v2;

static_assert(std::is_same<decltype(simd_cast<int32_t>(native_simd<int32_t>())),
                           native_simd<int32_t>>::value,
              "");

static_assert(
    std::is_same<decltype(simd_cast<int64_t>(fixed_size_simd<int32_t, 4>())),
                 fixed_size_simd<int64_t, 4>>::value,
    "");

static_assert(std::is_same<decltype(simd_cast<fixed_size_simd<int64_t, 1>>(
                               simd<int32_t, simd_abi::scalar>())),
                           fixed_size_simd<int64_t, 1>>::value,
              "");

static_assert(std::is_same<decltype(simd_cast<simd<int64_t, simd_abi::scalar>>(
                               fixed_size_simd<int32_t, 1>())),
                           simd<int64_t, simd_abi::scalar>>::value,
              "");

int main() {}
