// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -verify %s

typedef unsigned char v4u8 __attribute__((ext_vector_type(4)));
typedef unsigned short v4u16 __attribute__((ext_vector_type(4)));
typedef _Bool v4bool __attribute__((ext_vector_type(4)));
typedef float v4f32 __attribute__((ext_vector_type(4)));
typedef _Float16 v4f16 __attribute__((ext_vector_type(4)));
typedef unsigned char g4u8 __attribute__((vector_size(4)));
typedef float g4f32 __attribute__((vector_size(16)));

enum __attribute__((packed)) byte_enum {
  BYTE_ZERO,
};

_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m2_f16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m2_bf16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m2_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f64), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_f16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_bf16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3fn_f64), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_f16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_bf16), "");
_Static_assert(
    __has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m3fnu_f64), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e3m2fn_f16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e3m2fn_bf16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e3m2fn_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e3m2fn_f64), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e2m3fn_f16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e2m3fn_bf16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e2m3fn_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f6e2m3fn_f64), "");
// FP4 is deferred: _BitInt(4) vectors have no coherent memory layout yet.
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f4e2m1fn_f16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f4e2m1fn_bf16), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f4e2m1fn_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f4e2m1fn_f64), "");

_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2fnuz_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3fnuz_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e4m3b11fnuz_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e3m4_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e8m0fnu_f32), "");
_Static_assert(
    !__has_builtin(__builtin_elementwise_convert_from_f8e5m2_f80), "");
_Static_assert(!__has_builtin(__builtin_convert_from_arbitrary_fp), "");

void test_result_types(unsigned char b, v4u8 vb, g4u8 gb) {
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_f16(b)), _Float16), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_bf16(b)), __bf16), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_f32(b)), float), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_f32(vb)), v4f32), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e5m2_f32(gb)), g4f32), "");
  _Static_assert(__builtin_types_compatible_p(
      typeof(__builtin_elementwise_convert_from_f8e4m3fn_f16(vb)), v4f16), "");
}

void test_source_formats(unsigned char b) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b);
  (void)__builtin_elementwise_convert_from_f8e4m3fn_f32(b);
  (void)__builtin_elementwise_convert_from_f8e5m3fnu_f32(b);
  (void)__builtin_elementwise_convert_from_f8e5m2_f32((signed char)b);
  (void)__builtin_elementwise_convert_from_f8e5m2_f32((unsigned _BitInt(8))b);
}

// Integer promotions do not apply to the source.
void test_no_promotion(unsigned char b) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32((unsigned char)(b >> 1));
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b >> 1); // expected-error {{argument type 'int' must be exactly 8 bits wide to hold an 'f8e5m2' encoding}}
}

void test_arity(unsigned char b) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(); // expected-error {{too few arguments}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b, b); // expected-error {{too many arguments}}
}

void test_width(unsigned short b16, unsigned _BitInt(4) b4, v4u16 vb16) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b16); // expected-error {{argument type 'unsigned short' must be exactly 8 bits wide to hold an 'f8e5m2' encoding}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b4); // expected-error {{argument type 'unsigned _BitInt(4)' must be exactly 8 bits wide to hold an 'f8e5m2' encoding}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(vb16); // expected-error {{vector element type 'unsigned short' must be exactly 8 bits wide to hold an 'f8e5m2' encoding}}
}

void test_operand_types(float f, void *p) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(f); // expected-error {{argument of type 'float' cannot hold an 'f8e5m2' encoding; expected an integer of exactly 8 bits, a vector of such integers, or __mfp8}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(p); // expected-error {{argument of type 'void *' cannot hold an 'f8e5m2' encoding; expected an integer of exactly 8 bits, a vector of such integers, or __mfp8}}
}

void test_disallowed_integer_types(_Bool b, enum byte_enum e, v4bool vb) {
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(b); // expected-error {{argument of type '_Bool' cannot hold an 'f8e5m2' encoding}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(e); // expected-error {{argument of type 'enum byte_enum' cannot hold an 'f8e5m2' encoding}}
  (void)__builtin_elementwise_convert_from_f8e5m2_f32(vb); // expected-error {{cannot hold an 'f8e5m2' encoding}}
}

void test_volatile_source(volatile unsigned char *b) {
  __builtin_assume(
      __builtin_elementwise_convert_from_f8e5m2_f32(*b)); // expected-warning {{assumption is ignored because it contains (potential) side-effects}}
}
