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
// template <class T> struct is_simd_flag_type;
// template <class T> inline constexpr bool is_simd_flag_type_v = is_simd_flag_type<T>::value;

#include <cstdint>
#include <experimental/simd>
#include "test_macros.h"

using namespace std::experimental::parallelism_v2;

struct UserType {};

static_assert( is_simd_flag_type<element_aligned_tag>::value, "");
static_assert( is_simd_flag_type<vector_aligned_tag>::value, "");
static_assert( is_simd_flag_type<overaligned_tag<16>>::value, "");
static_assert( is_simd_flag_type<overaligned_tag<32>>::value, "");

static_assert(!is_simd_flag_type<void>::value, "");
static_assert(!is_simd_flag_type<int>::value, "");
static_assert(!is_simd_flag_type<float>::value, "");
static_assert(!is_simd_flag_type<UserType>::value, "");
static_assert(!is_simd_flag_type<simd<int8_t>>::value, "");
static_assert(!is_simd_flag_type<simd_mask<int8_t>>::value, "");

#if TEST_STD_VER > 14 && !defined(_LIBCPP_HAS_NO_VARIABLE_TEMPLATES) &&        \
    !defined(_LIBCPP_HAS_NO_INLINE_VARIABLES)

static_assert( is_simd_flag_type_v<element_aligned_tag>, "");
static_assert( is_simd_flag_type_v<vector_aligned_tag>, "");
static_assert( is_simd_flag_type_v<overaligned_tag<16>>, "");
static_assert( is_simd_flag_type_v<overaligned_tag<32>>, "");

static_assert(!is_simd_flag_type_v<void>, "");
static_assert(!is_simd_flag_type_v<int>, "");
static_assert(!is_simd_flag_type_v<float>, "");
static_assert(!is_simd_flag_type_v<UserType>, "");
static_assert(!is_simd_flag_type_v<simd<int8_t>>, "");
static_assert(!is_simd_flag_type_v<simd_mask<int8_t>>, "");

#endif

int main() {}
