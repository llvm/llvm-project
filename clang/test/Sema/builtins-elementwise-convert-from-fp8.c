// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fenable-matrix -verify %s

#define CHECK_BUILTIN(NAME) \
  static_assert(__has_builtin(NAME), #NAME); \
  static_assert(!__has_constexpr_builtin(NAME), #NAME)

CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m2_f16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m2_bf16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m2_f32);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e4m3fn_f16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e4m3fn_bf16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e4m3fn_f32);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m3fnu_f16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m3fnu_bf16);
CHECK_BUILTIN(__builtin_elementwise_convert_from_f8e5m3fnu_f32);

typedef unsigned char uchar;
typedef signed _BitInt(8) int8;
typedef unsigned _BitInt(8) uint8;

typedef char char1 __attribute__((ext_vector_type(1)));
typedef signed char schar2 __attribute__((ext_vector_type(2)));
typedef uchar uchar3 __attribute__((ext_vector_type(3)));
typedef uchar uchar4 __attribute__((ext_vector_type(4)));
typedef int8 int8x4 __attribute__((ext_vector_type(4)));
typedef uint8 uint8x4 __attribute__((ext_vector_type(4)));
typedef _Float16 half2 __attribute__((ext_vector_type(2)));
typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef __bf16 bfloat3 __attribute__((ext_vector_type(3)));
typedef __bf16 bfloat4 __attribute__((ext_vector_type(4)));
typedef float float1 __attribute__((ext_vector_type(1)));
typedef float float4 __attribute__((ext_vector_type(4)));

typedef uchar gnu_uchar4 __attribute__((vector_size(4)));
typedef int8 gnu_int8x4 __attribute__((vector_size(4)));
typedef _Float16 gnu_half4 __attribute__((vector_size(8)));
typedef __bf16 gnu_bfloat4 __attribute__((vector_size(8)));
typedef float gnu_float4 __attribute__((vector_size(16)));

#define CHECK_TYPE(EXPR, TYPE) \
  static_assert(__builtin_types_compatible_p(__typeof__(EXPR), TYPE), "incorrect result type")

#define CHECK_FORMAT(FORMAT, BITS, HALF, BFLOAT, FLOAT) \
  CHECK_TYPE(__builtin_elementwise_convert_from_##FORMAT##_f16(BITS), HALF); \
  CHECK_TYPE(__builtin_elementwise_convert_from_##FORMAT##_bf16(BITS), BFLOAT); \
  CHECK_TYPE(__builtin_elementwise_convert_from_##FORMAT##_f32(BITS), FLOAT)

void scalar_types(char c, signed char sc, uchar uc, int8 i8, uint8 u8,
                  const uchar cc, volatile uchar vc) {
  CHECK_FORMAT(f8e5m2, c, _Float16, __bf16, float);
  CHECK_FORMAT(f8e4m3fn, sc, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m3fnu, uc, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m2, i8, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m2, u8, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m2, cc, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m2, vc, _Float16, __bf16, float);
  CHECK_FORMAT(f8e5m2, (uchar)(uc >> 1), _Float16, __bf16, float);
}

void vector_types(char1 v1, schar2 v2, uchar3 v3, uchar4 v4,
                  int8x4 i8, uint8x4 u8, const uchar4 cv,
                  gnu_uchar4 gv, gnu_int8x4 gi8) {
  CHECK_FORMAT(f8e5m2, v4, half4, bfloat4, float4);
  CHECK_FORMAT(f8e4m3fn, v4, half4, bfloat4, float4);
  CHECK_FORMAT(f8e5m3fnu, v4, half4, bfloat4, float4);
  CHECK_FORMAT(f8e5m2, i8, half4, bfloat4, float4);
  CHECK_FORMAT(f8e5m2, u8, half4, bfloat4, float4);
  CHECK_FORMAT(f8e5m2, cv, half4, bfloat4, float4);
  CHECK_FORMAT(f8e5m2, gv, gnu_half4, gnu_bfloat4, gnu_float4);
  CHECK_FORMAT(f8e4m3fn, gv, gnu_half4, gnu_bfloat4, gnu_float4);
  CHECK_FORMAT(f8e5m3fnu, gv, gnu_half4, gnu_bfloat4, gnu_float4);
  CHECK_FORMAT(f8e5m2, gi8, gnu_half4, gnu_bfloat4, gnu_float4);

  // Element counts are preserved, including one and non-power-of-two counts.
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m2_f32(v1), float1);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e4m3fn_f16(v2), half2);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m3fnu_bf16(v3), bfloat3);
  CHECK_TYPE(__builtin_elementwise_convert_from_f8e5m2_f32(v4.x), float);
}

typedef short short4 __attribute__((ext_vector_type(4)));
typedef bool bool8 __attribute__((ext_vector_type(8)));
typedef unsigned _BitInt(4) uint4x4 __attribute__((ext_vector_type(4)));
typedef uchar uchar2x2 __attribute__((matrix_type(2, 2)));
enum byte_enum : unsigned char { byte_zero };
struct bytes {
  uchar value;
};

void invalid_types(bool b, enum byte_enum e, short s, int i, long l,
                   _BitInt(7) i7, _BitInt(9) i9, unsigned _BitInt(1) u1,
                   float f, __bf16 bf, _Complex float complex,
                   uchar *p, struct bytes aggregate,
                   short4 sv, float4 fv, bool8 bv, uint4x4 u4v, uchar2x2 m) {
  uchar a[4];
  __builtin_elementwise_convert_from_f8e5m2_f32(b);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(e);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(s);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(i);
  // expected-error@-1 {{argument to '__builtin_elementwise_convert_from_f8e5m2_f32' must be an 8-bit integer or a vector of 8-bit integers (was 'int')}}
  __builtin_elementwise_convert_from_f8e5m2_f32(l);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(i7);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(i9);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(u1);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(f);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(bf);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(complex);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(p);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(a);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(aggregate);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(sv);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(fv);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(bv);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(u4v);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(m);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
}

void promotions(uchar bits) {
  __builtin_elementwise_convert_from_f8e5m2_f32(0x38);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(bits >> 1);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(+bits);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
}

void arity(uchar bits) {
  __builtin_elementwise_convert_from_f8e5m2_f16();
  // expected-error@-1 {{too few arguments to function call, expected 1, have 0}}
  __builtin_elementwise_convert_from_f8e4m3fn_bf16(bits, bits);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 2}}
  __builtin_elementwise_convert_from_f8e5m3fnu_f32(bits, bits, bits);
  // expected-error@-1 {{too many arguments to function call, expected 1, have 3}}
}

// These calls cannot be used in constant expressions, including initializers
// for objects with static storage duration in C.
_Float16 global_half = __builtin_elementwise_convert_from_f8e5m2_f16((uchar)0);
// expected-error@-1 {{initializer element is not a compile-time constant}}
__bf16 global_bfloat = __builtin_elementwise_convert_from_f8e4m3fn_bf16((uchar)0);
// expected-error@-1 {{initializer element is not a compile-time constant}}
float global_float = __builtin_elementwise_convert_from_f8e5m3fnu_f32((uchar)0);
// expected-error@-1 {{initializer element is not a compile-time constant}}

void static_initializer(void) {
  static float4 values = __builtin_elementwise_convert_from_f8e5m2_f32((uchar4){0});
  // expected-error@-1 {{initializer element is not a compile-time constant}}
}
