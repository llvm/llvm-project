// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c23 -fexperimental-overflow-behavior-types -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x c++ -std=c++20 -fexperimental-overflow-behavior-types -verify %s

typedef unsigned char __attribute__((overflow_behavior(wrap))) wrap_uchar;
typedef signed _BitInt(8) __attribute__((overflow_behavior(trap))) trap_int8;
typedef unsigned _BitInt(7) __attribute__((overflow_behavior(wrap))) wrap_uint7;
typedef unsigned _BitInt(1) __attribute__((overflow_behavior(trap))) trap_uint1;
typedef bool __attribute__((overflow_behavior(wrap))) wrap_bool;
enum byte_enum : unsigned char { byte_zero };
typedef enum byte_enum __attribute__((overflow_behavior(trap))) trap_enum;

typedef const volatile wrap_uchar qualified_byte;
typedef const volatile wrap_uint7 qualified_uint7;
typedef qualified_byte byte_alias;
typedef qualified_uint7 uint7_alias;

void valid_inputs(byte_alias *c, const trap_int8 i) {
  // Overflow behavior does not change the source encoding or result type.
  _Float16 h = __builtin_elementwise_convert_from_f8e5m2_f16(*c);
  __bf16 b = __builtin_elementwise_convert_from_f8e4m3fn_bf16(i);
  float f = __builtin_elementwise_convert_from_f8e5m3fnu_f32(i);
}

void invalid_inputs(uint7_alias *u7, const trap_uint1 u1,
                    const wrap_bool b, const trap_enum e) {
  // Check the underlying integer's width, not the wrapper's storage size.
  __builtin_elementwise_convert_from_f8e5m2_f32(*u7);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m2_f32(u1);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}

  // Wrapping a boolean or enumeration must not hide its type category.
  __builtin_elementwise_convert_from_f8e4m3fn_f16(b);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
  __builtin_elementwise_convert_from_f8e5m3fnu_bf16(e);
  // expected-error@-1 {{must be an 8-bit integer or a vector of 8-bit integers}}
}

#ifdef __cplusplus
template <class T>
concept can_convert = requires(T bits) {
  __builtin_elementwise_convert_from_f8e5m2_f32(bits);
};
static_assert(can_convert<wrap_uchar>);
static_assert(can_convert<const trap_int8 &>);
static_assert(!can_convert<qualified_uint7>);
static_assert(!can_convert<wrap_bool>);
static_assert(!can_convert<trap_enum>);
#endif
