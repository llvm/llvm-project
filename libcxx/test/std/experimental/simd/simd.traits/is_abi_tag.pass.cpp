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
// template <class T> struct is_abi_tag;
// template <class T> inline constexpr bool is_abi_tag_v = is_abi_tag<T>::value;

#include <cstdint>
#include <experimental/simd>
#include "test_macros.h"

using namespace std::experimental::parallelism_v2;

struct UserType {};

static_assert( is_abi_tag<simd_abi::native<int8_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<int16_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<int32_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<int64_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<uint8_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<uint16_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<uint32_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<uint64_t>>::value, "");
static_assert( is_abi_tag<simd_abi::native<float>>::value, "");
static_assert( is_abi_tag<simd_abi::native<double>>::value, "");

static_assert( is_abi_tag<simd_abi::compatible<int8_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<int16_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<int32_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<int64_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<uint8_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<uint16_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<uint32_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<uint64_t>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<float>>::value, "");
static_assert( is_abi_tag<simd_abi::compatible<double>>::value, "");

static_assert( is_abi_tag<simd_abi::scalar>::value, "");
static_assert(!std::is_same<simd_abi::scalar, simd_abi::fixed_size<1>>::value,
              "");

static_assert( is_abi_tag<simd_abi::fixed_size<1>>::value, "");
static_assert( is_abi_tag<simd_abi::fixed_size<2>>::value, "");
static_assert( is_abi_tag<simd_abi::fixed_size<3>>::value, "");
static_assert( is_abi_tag<simd_abi::fixed_size<4>>::value, "");
static_assert( is_abi_tag<simd_abi::fixed_size<5>>::value, "");
static_assert( is_abi_tag<simd_abi::fixed_size<32>>::value, "");

static_assert(!is_abi_tag<void>::value, "");
static_assert(!is_abi_tag<int>::value, "");
static_assert(!is_abi_tag<float>::value, "");
static_assert(!is_abi_tag<UserType>::value, "");
static_assert(!is_abi_tag<simd<int>>::value, "");
static_assert(!is_abi_tag<simd<float>>::value, "");
static_assert(!is_abi_tag<simd_mask<int>>::value, "");
static_assert(!is_abi_tag<simd_mask<float>>::value, "");

#if TEST_STD_VER > 14 && !defined(_LIBCPP_HAS_NO_VARIABLE_TEMPLATES) &&        \
    !defined(_LIBCPP_HAS_NO_INLINE_VARIABLES)

static_assert( is_abi_tag_v<simd_abi::native<int8_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<int16_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<int32_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<int64_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<uint8_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<uint16_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<uint32_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<uint64_t>>, "");
static_assert( is_abi_tag_v<simd_abi::native<float>>, "");
static_assert( is_abi_tag_v<simd_abi::native<double>>, "");

static_assert( is_abi_tag_v<simd_abi::compatible<int8_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<int16_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<int32_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<int64_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<uint8_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<uint16_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<uint32_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<uint64_t>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<float>>, "");
static_assert( is_abi_tag_v<simd_abi::compatible<double>>, "");

static_assert( is_abi_tag_v<simd_abi::scalar>, "");
static_assert(!std::is_same<simd_abi::scalar, simd_abi::fixed_size<1>>::value,
              "");

static_assert( is_abi_tag_v<simd_abi::fixed_size<1>>, "");
static_assert( is_abi_tag_v<simd_abi::fixed_size<2>>, "");
static_assert( is_abi_tag_v<simd_abi::fixed_size<3>>, "");
static_assert( is_abi_tag_v<simd_abi::fixed_size<4>>, "");
static_assert( is_abi_tag_v<simd_abi::fixed_size<5>>, "");
static_assert( is_abi_tag_v<simd_abi::fixed_size<32>>, "");

static_assert(!is_abi_tag_v<void>, "");
static_assert(!is_abi_tag_v<int>, "");
static_assert(!is_abi_tag_v<float>, "");
static_assert(!is_abi_tag_v<UserType>, "");
static_assert(!is_abi_tag_v<simd<int>>, "");
static_assert(!is_abi_tag_v<simd<float>>, "");
static_assert(!is_abi_tag_v<simd_mask<int>>, "");
static_assert(!is_abi_tag_v<simd_mask<float>>, "");

#endif

int main() {}
