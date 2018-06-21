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
// [simd.traits]
// template <class T, size_t N> struct abi_for_size { using type = see below ; };
// template <class T, size_t N> using abi_for_size_t = typename abi_for_size<T, N>::type;

#include <cstdint>
#include <experimental/simd>

using namespace std::experimental::parallelism_v2;

static_assert(std::is_same<typename abi_for_size<int, 4>::type,
                           simd_abi::fixed_size<4>>::value,
              "");

static_assert(
    std::is_same<abi_for_size_t<int, 4>, simd_abi::fixed_size<4>>::value, "");

int main() {}
