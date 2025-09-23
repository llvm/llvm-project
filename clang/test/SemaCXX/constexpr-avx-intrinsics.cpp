// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -target-feature +avx -target-feature +avx2 \
// RUN:   -fsyntax-only -verify %s
// expected-no-diagnostics

// 128/256bit vector type

using v2i64 = long long __attribute__((vector_size(16)));  // 2 x i64 = 128b
using v4i64 = long long __attribute__((vector_size(32)));  // 4 x i64 = 256b
using v4f32 = float      __attribute__((vector_size(16)));  // 4 x f32 = 128b
using v8f32 = float      __attribute__((vector_size(32)));  // 8 x f32 = 256b
using v2f64 = double     __attribute__((vector_size(16)));  // 2 x f64 = 128b
using v4f64 = double     __attribute__((vector_size(32)));  // 4 x f64 = 256b
using v4i32 = int        __attribute__((vector_size(16)));  // 4 x i32 = 128b
using v8i32 = int        __attribute__((vector_size(32)));  // 8 x i32 = 256b


// source vectors (constexpr)
constexpr v4i64 SRC_I64_256 = {10, 20, 30, 40};
constexpr v8f32 SRC_F32_256 = {0, 1, 2, 3, 4, 5, 6, 7};
constexpr v4f64 SRC_F64_256 = {1.1, 2.2, 3.3, 4.4};
constexpr v8i32 SRC_I32_256 = {10, 20, 30, 40, 50, 60, 70, 80};

// 1) __builtin_ia32_extract128i256 : 256비트 i64 벡터 -> 하위/상위 128비트 추출
constexpr v2i64 R_EXTRACT_I128_0 = __builtin_ia32_extract128i256(SRC_I64_256, 0);
static_assert(R_EXTRACT_I128_0[0] == 10 && R_EXTRACT_I128_0[1] == 20);

constexpr v2i64 R_EXTRACT_I128_1 = __builtin_ia32_extract128i256(SRC_I64_256, 1);
static_assert(R_EXTRACT_I128_1[0] == 30 && R_EXTRACT_I128_1[1] == 40);

// // 2) __builtin_ia32_vextractf128_ps256 : 256비트 f32 -> 128비트 f32
// constexpr v4f32 R_EXTRACT_F128_PS_0 = __builtin_ia32_vextractf128_ps256(SRC_F32_256, 0);
// static_assert(R_EXTRACT_F128_PS_0[0] == 0 && R_EXTRACT_F128_PS_0[1] == 1 &&
//               R_EXTRACT_F128_PS_0[2] == 2 && R_EXTRACT_F128_PS_0[3] == 3);

// constexpr v4f32 R_EXTRACT_F128_PS_1 = __builtin_ia32_vextractf128_ps256(SRC_F32_256, 1);
// static_assert(R_EXTRACT_F128_PS_1[0] == 4 && R_EXTRACT_F128_PS_1[1] == 5 &&
//               R_EXTRACT_F128_PS_1[2] == 6 && R_EXTRACT_F128_PS_1[3] == 7);

// // 3) __builtin_ia32_vextractf128_pd256 : 256비트 f64 -> 128비트 f64
// constexpr v2f64 R_EXTRACT_F128_PD_0 = __builtin_ia32_vextractf128_pd256(SRC_F64_256, 0);
// static_assert(R_EXTRACT_F128_PD_0[0] == 1.1 && R_EXTRACT_F128_PD_0[1] == 2.2);

// constexpr v2f64 R_EXTRACT_F128_PD_1 = __builtin_ia32_vextractf128_pd256(SRC_F64_256, 1);
// static_assert(R_EXTRACT_F128_PD_1[0] == 3.3 && R_EXTRACT_F128_PD_1[1] == 4.4);

// // 4) __builtin_ia32_vextractf128_si256 : 256비트 i32 -> 128비트 i32
// constexpr v4i32 R_EXTRACT_F128_SI256_0 = __builtin_ia32_vextractf128_si256(SRC_I32_256, 0);
// static_assert(R_EXTRACT_F128_SI256_0[0] == 10 && R_EXTRACT_F128_SI256_0[1] == 20 &&
//               R_EXTRACT_F128_SI256_0[2] == 30 && R_EXTRACT_F128_SI256_0[3] == 40);

// constexpr v4i32 R_EXTRACT_F128_SI256_1 = __builtin_ia32_vextractf128_si256(SRC_I32_256, 1);
// static_assert(R_EXTRACT_F128_SI256_1[0] == 50 && R_EXTRACT_F128_SI256_1[1] == 60 &&
//               R_EXTRACT_F128_SI256_1[2] == 70 && R_EXTRACT_F128_SI256_1[3] == 80);