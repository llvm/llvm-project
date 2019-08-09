// RUN: %clang_cc1 %s -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify

// This test is invalid under the sizeless type extension and is a stop-gap
// until that extension is added.  The test makes sure that sizeof and
// alignof queries are handled without assertion failures, since at
// present there is nothing to prevent such queries being made.
//
// Under this scheme, sizeof returns 0 for all built-in sizeless types.
// This is compatible with correct usage but it relies on the user being
// careful to avoid constructs that depend directly or indirectly on the
// value of sizeof.  (The sizeless type extension avoids this by treating
// such constructs as an error.)

// expected-no-diagnostics

void f() {
  int size_s8[sizeof(__SVInt8_t) == 0 ? 1 : -1];
  int align_s8[__alignof__(__SVInt8_t) == 16 ? 1 : -1];

  int size_s16[sizeof(__SVInt16_t) == 0 ? 1 : -1];
  int align_s16[__alignof__(__SVInt16_t) == 16 ? 1 : -1];

  int size_s32[sizeof(__SVInt32_t) == 0 ? 1 : -1];
  int align_s32[__alignof__(__SVInt32_t) == 16 ? 1 : -1];

  int size_s64[sizeof(__SVInt64_t) == 0 ? 1 : -1];
  int align_s64[__alignof__(__SVInt64_t) == 16 ? 1 : -1];

  int size_u8[sizeof(__SVUint8_t) == 0 ? 1 : -1];
  int align_u8[__alignof__(__SVUint8_t) == 16 ? 1 : -1];

  int size_u16[sizeof(__SVUint16_t) == 0 ? 1 : -1];
  int align_u16[__alignof__(__SVUint16_t) == 16 ? 1 : -1];

  int size_u32[sizeof(__SVUint32_t) == 0 ? 1 : -1];
  int align_u32[__alignof__(__SVUint32_t) == 16 ? 1 : -1];

  int size_u64[sizeof(__SVUint64_t) == 0 ? 1 : -1];
  int align_u64[__alignof__(__SVUint64_t) == 16 ? 1 : -1];

  int size_f16[sizeof(__SVFloat16_t) == 0 ? 1 : -1];
  int align_f16[__alignof__(__SVFloat16_t) == 16 ? 1 : -1];

  int size_f32[sizeof(__SVFloat32_t) == 0 ? 1 : -1];
  int align_f32[__alignof__(__SVFloat32_t) == 16 ? 1 : -1];

  int size_f64[sizeof(__SVFloat64_t) == 0 ? 1 : -1];
  int align_f64[__alignof__(__SVFloat64_t) == 16 ? 1 : -1];

  int size_b8[sizeof(__SVBool_t) == 0 ? 1 : -1];
  int align_b8[__alignof__(__SVBool_t) == 2 ? 1 : -1];
}
