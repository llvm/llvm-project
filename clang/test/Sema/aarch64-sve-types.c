// RUN: %clang_cc1 %s -triple aarch64-none-linux-gnu -target-feature +sve,+bf16 -fsyntax-only -verify

void f() {
  int size_s8[sizeof(__SVInt8_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVInt8_t'}}
  int align_s8[__alignof__(__SVInt8_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVInt8_t'}}

  int size_s16[sizeof(__SVInt16_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVInt16_t'}}
  int align_s16[__alignof__(__SVInt16_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVInt16_t'}}

  int size_s32[sizeof(__SVInt32_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVInt32_t'}}
  int align_s32[__alignof__(__SVInt32_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVInt32_t'}}

  int size_s64[sizeof(__SVInt64_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVInt64_t'}}
  int align_s64[__alignof__(__SVInt64_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVInt64_t'}}

  int size_u8[sizeof(__SVUint8_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVUint8_t'}}
  int align_u8[__alignof__(__SVUint8_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVUint8_t'}}

  int size_u16[sizeof(__SVUint16_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVUint16_t'}}
  int align_u16[__alignof__(__SVUint16_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVUint16_t'}}

  int size_u32[sizeof(__SVUint32_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVUint32_t'}}
  int align_u32[__alignof__(__SVUint32_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVUint32_t'}}

  int size_u64[sizeof(__SVUint64_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVUint64_t'}}
  int align_u64[__alignof__(__SVUint64_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVUint64_t'}}

  int size_f16[sizeof(__SVFloat16_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVFloat16_t'}}
  int align_f16[__alignof__(__SVFloat16_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVFloat16_t'}}

  int size_f32[sizeof(__SVFloat32_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVFloat32_t'}}
  int align_f32[__alignof__(__SVFloat32_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVFloat32_t'}}

  int size_f64[sizeof(__SVFloat64_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVFloat64_t'}}
  int align_f64[__alignof__(__SVFloat64_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVFloat64_t'}}

  int size_bf16[sizeof(__SVBFloat16_t) == 0 ? 1 : -1];        // expected-error {{invalid application of 'sizeof' to sizeless type '__SVBFloat16_t'}}
  int align_bf16[__alignof__(__SVBFloat16_t) == 16 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVBFloat16_t'}}

  int size_b8[sizeof(__SVBool_t) == 0 ? 1 : -1];       // expected-error {{invalid application of 'sizeof' to sizeless type '__SVBool_t'}}
  int align_b8[__alignof__(__SVBool_t) == 2 ? 1 : -1]; // expected-error {{invalid application of '__alignof' to sizeless type '__SVBool_t'}}
}
