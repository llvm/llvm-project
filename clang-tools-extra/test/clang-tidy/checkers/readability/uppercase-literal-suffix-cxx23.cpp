// TODO: When Clang adds support for C++23 floating-point types, enable these tests by:
//    1. Removing all the #if 0 + #endif guards.
//    2. Removing all occurrences of the string "DISABLED-" in this file.
//    3. Deleting this message.
// These suffixes may be relevant to C too: https://github.com/llvm/llvm-project/issues/97335

// RUN: %check_clang_tidy -std=c++23-or-later %s readability-uppercase-literal-suffix %t -- -- -target aarch64-linux-gnu -I %clang_tidy_headers

#include "integral_constant.h"
#include <cstddef>
#if 0
#include <stdfloat>
#endif

void normal_literals() {
  // std::bfloat16_t

#if 0
  static constexpr auto v1 = 1.bf16;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'bf16', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v1 = 1.BF16;
  static_assert(is_same<decltype(v1), const std::bfloat16_t>::value, "");
  static_assert(v1 == 1.BF16, "");

  static constexpr auto v2 = 1.e0bf16;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'bf16', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v2 = 1.e0BF16;
  static_assert(is_same<decltype(v2), const std::bfloat16_t>::value, "");
  static_assert(v2 == 1.BF16, "");

  static constexpr auto v3 = 1.BF16; // OK.
  static_assert(is_same<decltype(v3), const std::bfloat16_t>::value, "");
  static_assert(v3 == 1.BF16, "");

  static constexpr auto v4 = 1.e0BF16; // OK.
  static_assert(is_same<decltype(v4), const std::bfloat16_t>::value, "");
  static_assert(v4 == 1.BF16, "");
#endif

  // _Float16/std::float16_t

  static constexpr auto v5 = 1.f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-FIXES: static constexpr auto v5 = 1.F16;
  static_assert(is_same<decltype(v5), const _Float16>::value, "");
  static_assert(v5 == 1.F16, "");

  static constexpr auto v6 = 1.e0f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-FIXES: static constexpr auto v6 = 1.e0F16;
  static_assert(is_same<decltype(v6), const _Float16>::value, "");
  static_assert(v6 == 1.F16, "");

  static constexpr auto v7 = 1.F16; // OK.
  static_assert(is_same<decltype(v7), const _Float16>::value, "");
  static_assert(v7 == 1.F16, "");

  static constexpr auto v8 = 1.e0F16; // OK.
  static_assert(is_same<decltype(v8), const _Float16>::value, "");
  static_assert(v8 == 1.F16, "");

  // std::float32_t

#if 0
  static constexpr auto v9 = 1.f32;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f32', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v9 = 1.F32;
  static_assert(is_same<decltype(v9), const std::float32_t>::value, "");
  static_assert(v9 == 1.F32, "");

  static constexpr auto v10 = 1.e0f32;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f32', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v10 = 1.e0F32;
  static_assert(is_same<decltype(v10), const std::float32_t>::value, "");
  static_assert(v10 == 1.F32, "");

  static constexpr auto v11 = 1.F32; // OK.
  static_assert(is_same<decltype(v11), const std::float32_t>::value, "");
  static_assert(v11 == 1.F32, "");

  static constexpr auto v12 = 1.e0F32; // OK.
  static_assert(is_same<decltype(v12), const std::float32_t>::value, "");
  static_assert(v12 == 1.F32, "");
#endif

  // std::float64_t

#if 0
  static constexpr auto v13 = 1.f64;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f64', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v13 = 1.F64;
  static_assert(is_same<decltype(v13), const std::float64_t>::value, "");
  static_assert(v13 == 1.F64, "");

  static constexpr auto v14 = 1.e0f64;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f64', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v14 = 1.e0F64;
  static_assert(is_same<decltype(v14), const std::float64_t>::value, "");
  static_assert(v14 == 1.F64, "");

  static constexpr auto v15 = 1.F64; // OK.
  static_assert(is_same<decltype(v15), const std::float64_t>::value, "");
  static_assert(v15 == 1.F64, "");

  static constexpr auto v16 = 1.e0F64; // OK.
  static_assert(is_same<decltype(v16), const std::float64_t>::value, "");
  static_assert(v16 == 1.F64, "");
#endif

  // std::float128_t

#if 0
  static constexpr auto v17 = 1.f128;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f128', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v17 = 1.F128;
  static_assert(is_same<decltype(v17), const std::float128_t>::value, "");
  static_assert(v17 == 1.F128, "");

  static constexpr auto v18 = 1.e0f128;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f128', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v18 = 1.e0F128;
  static_assert(is_same<decltype(v18), const std::float128_t>::value, "");
  static_assert(v18 == 1.F128, "");

  static constexpr auto v19 = 1.F128; // OK.
  static_assert(is_same<decltype(v19), const std::float128_t>::value, "");
  static_assert(v19 == 1.F128, "");

  static constexpr auto v20 = 1.e0F128; // OK.
  static_assert(is_same<decltype(v20), const std::float128_t>::value, "");
  static_assert(v20 == 1.F128, "");
#endif
}

void hexadecimal_literals() {
  // std::bfloat16_t

#if 0
  static constexpr auto v1 = 0xfp0bf16;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'bf16', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v1 = 0xfp0BF16;
  static_assert(is_same<decltype(v1), const std::bfloat16_t>::value, "");
  static_assert(v1 == 0xfp0BF16, "");

  static constexpr auto v2 = 0xfp0BF16; // OK.
  static_assert(is_same<decltype(v2), const std::bfloat16_t>::value, "");
  static_assert(v2 == 0xfp0BF16, "");
#endif

  // _Float16/std::float16_t

  static constexpr auto v3 = 0xfp0f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-FIXES: static constexpr auto v3 = 0xfp0F16;
  static_assert(is_same<decltype(v3), const _Float16>::value, "");
  static_assert(v3 == 0xfp0F16, "");

  static constexpr auto v4 = 0xfp0F16; // OK.
  static_assert(is_same<decltype(v4), const _Float16>::value, "");
  static_assert(v4 == 0xfp0F16, "");

  // std::float32_t

#if 0
  static constexpr auto v5 = 0xfp0f32;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f32', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v5 = 0xfp0F32;
  static_assert(is_same<decltype(v5), const std::float32_t>::value, "");
  static_assert(v5 == 0xfp0F32, "");

  static constexpr auto v6 = 0xfp0F32; // OK.
  static_assert(is_same<decltype(v6), const std::float32_t>::value, "");
  static_assert(v6 == 0xfp0F32, "");
#endif

  // std::float64_t

#if 0
  static constexpr auto v7 = 0xfp0f64;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f64', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v7 = 0xfp0F64;
  static_assert(is_same<decltype(v7), const std::float64_t>::value, "");
  static_assert(v7 == 0xfp0F64, "");

  static constexpr auto v8 = 0xfp0F64; // OK.
  static_assert(is_same<decltype(v8), const std::float64_t>::value, "");
  static_assert(v8 == 0xfp0F64, "");
#endif

  // std::float128_t

#if 0
  static constexpr auto v9 = 0xfp0f128;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f128', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v9 = 0xfp0F128;
  static_assert(is_same<decltype(v9), const std::float128_t>::value, "");
  static_assert(v9 == 0xfp0F128, "");

  static constexpr auto v10 = 0xfp0F128; // OK.
  static_assert(is_same<decltype(v10), const std::float128_t>::value, "");
  static_assert(v10 == 0xfp0F128, "");
#endif

}

void size_t_suffix() {
  // Signed

  static constexpr auto v29 = 1z;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'z', which is not uppercase
  // CHECK-FIXES: static constexpr auto v29 = 1Z;
  static_assert(v29 == 1Z, "");

  static constexpr auto v30 = 1Z; // OK.
  static_assert(v30 == 1Z, "");

  // size_t Unsigned

  static constexpr auto v31 = 1zu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'zu', which is not uppercase
  // CHECK-FIXES: static constexpr auto v31 = 1ZU;
  static_assert(is_same<decltype(v31), const size_t>::value, "");
  static_assert(v31 == 1ZU, "");

  static constexpr auto v32 = 1Zu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'Zu', which is not uppercase
  // CHECK-FIXES: static constexpr auto v32 = 1ZU;
  static_assert(is_same<decltype(v32), const size_t>::value, "");
  static_assert(v32 == 1ZU, "");

  static constexpr auto v33 = 1zU;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'zU', which is not uppercase
  // CHECK-FIXES: static constexpr auto v33 = 1ZU;
  static_assert(is_same<decltype(v33), const size_t>::value, "");
  static_assert(v33 == 1ZU, "");

  static constexpr auto v34 = 1ZU; // OK.
  static_assert(is_same<decltype(v34), const size_t>::value, "");
  static_assert(v34 == 1ZU, "");

  // Unsigned size_t

  static constexpr auto v35 = 1uz;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'uz', which is not uppercase
  // CHECK-FIXES: static constexpr auto v35 = 1UZ;
  static_assert(is_same<decltype(v35), const size_t>::value, "");
  static_assert(v35 == 1UZ);

  static constexpr auto v36 = 1uZ;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'uZ', which is not uppercase
  // CHECK-FIXES: static constexpr auto v36 = 1UZ;
  static_assert(is_same<decltype(v36), const size_t>::value, "");
  static_assert(v36 == 1UZ);

  static constexpr auto v37 = 1Uz;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'Uz', which is not uppercase
  // CHECK-FIXES: static constexpr auto v37 = 1UZ;
  static_assert(is_same<decltype(v37), const size_t>::value, "");
  static_assert(v37 == 1UZ);

  static constexpr auto v38 = 1UZ; // OK.
  static_assert(is_same<decltype(v38), const size_t>::value, "");
  static_assert(v38 == 1UZ);
}
