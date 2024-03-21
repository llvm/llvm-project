// RUN: %clang_cc1 -triple loongarch64 -target-feature +lsx -verify %s
// RUN: not %clang_cc1 -triple loongarch64 -DFEATURE_CHECK -emit-llvm %s -o /dev/null 2>&1 \
// RUN:   | FileCheck %s

typedef signed char v16i8 __attribute__((vector_size(16), aligned(16)));
typedef signed char v16i8_b __attribute__((vector_size(16), aligned(1)));
typedef unsigned char v16u8 __attribute__((vector_size(16), aligned(16)));
typedef unsigned char v16u8_b __attribute__((vector_size(16), aligned(1)));
typedef short v8i16 __attribute__((vector_size(16), aligned(16)));
typedef short v8i16_h __attribute__((vector_size(16), aligned(2)));
typedef unsigned short v8u16 __attribute__((vector_size(16), aligned(16)));
typedef unsigned short v8u16_h __attribute__((vector_size(16), aligned(2)));
typedef int v4i32 __attribute__((vector_size(16), aligned(16)));
typedef int v4i32_w __attribute__((vector_size(16), aligned(4)));
typedef unsigned int v4u32 __attribute__((vector_size(16), aligned(16)));
typedef unsigned int v4u32_w __attribute__((vector_size(16), aligned(4)));
typedef long long v2i64 __attribute__((vector_size(16), aligned(16)));
typedef long long v2i64_d __attribute__((vector_size(16), aligned(8)));
typedef unsigned long long v2u64 __attribute__((vector_size(16), aligned(16)));
typedef unsigned long long v2u64_d __attribute__((vector_size(16), aligned(8)));
typedef float v4f32 __attribute__((vector_size(16), aligned(16)));
typedef float v4f32_w __attribute__((vector_size(16), aligned(4)));
typedef double v2f64 __attribute__((vector_size(16), aligned(16)));
typedef double v2f64_d __attribute__((vector_size(16), aligned(8)));

typedef long long __m128i __attribute__((__vector_size__(16), __may_alias__));
typedef float __m128 __attribute__((__vector_size__(16), __may_alias__));
typedef double __m128d __attribute__((__vector_size__(16), __may_alias__));

#ifdef FEATURE_CHECK
void test_feature(v16i8 _1) {
// CHECK: error: '__builtin_lsx_vslli_b' needs target feature lsx
  (void)__builtin_lsx_vslli_b(_1, 1);
}
#endif

v16i8 vslli_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vslli_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vslli_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vslli_b(_1, var); // expected-error {{argument to '__builtin_lsx_vslli_b' must be a constant integer}}
  return res;
}

v8i16 vslli_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vslli_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vslli_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vslli_h(_1, var); // expected-error {{argument to '__builtin_lsx_vslli_h' must be a constant integer}}
  return res;
}

v4i32 vslli_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vslli_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslli_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslli_w(_1, var); // expected-error {{argument to '__builtin_lsx_vslli_w' must be a constant integer}}
  return res;
}

v2i64 vslli_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vslli_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vslli_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vslli_d(_1, var); // expected-error {{argument to '__builtin_lsx_vslli_d' must be a constant integer}}
  return res;
}

v16i8 vsrai_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsrai_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrai_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrai_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsrai_b' must be a constant integer}}
  return res;
}

v8i16 vsrai_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsrai_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrai_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrai_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsrai_h' must be a constant integer}}
  return res;
}

v4i32 vsrai_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsrai_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrai_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrai_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsrai_w' must be a constant integer}}
  return res;
}

v2i64 vsrai_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsrai_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrai_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrai_d(_1, var); // expected-error {{argument to '__builtin_lsx_vsrai_d' must be a constant integer}}
  return res;
}

v16i8 vsrari_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsrari_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrari_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrari_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsrari_b' must be a constant integer}}
  return res;
}

v8i16 vsrari_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsrari_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrari_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrari_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsrari_h' must be a constant integer}}
  return res;
}

v4i32 vsrari_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsrari_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrari_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrari_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsrari_w' must be a constant integer}}
  return res;
}

v2i64 vsrari_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsrari_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrari_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrari_d(_1, var); // expected-error {{argument to '__builtin_lsx_vsrari_d' must be a constant integer}}
  return res;
}

v16i8 vsrli_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsrli_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrli_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrli_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsrli_b' must be a constant integer}}
  return res;
}

v8i16 vsrli_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsrli_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrli_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrli_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsrli_h' must be a constant integer}}
  return res;
}

v4i32 vsrli_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsrli_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrli_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrli_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsrli_w' must be a constant integer}}
  return res;
}

v2i64 vsrli_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsrli_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrli_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrli_d(_1, var); // expected-error {{argument to '__builtin_lsx_vsrli_d' must be a constant integer}}
  return res;
}

v16i8 vsrlri_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsrlri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrlri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsrlri_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsrlri_b' must be a constant integer}}
  return res;
}

v8i16 vsrlri_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsrlri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlri_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsrlri_h' must be a constant integer}}
  return res;
}

v4i32 vsrlri_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsrlri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlri_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsrlri_w' must be a constant integer}}
  return res;
}

v2i64 vsrlri_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsrlri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlri_d(_1, var); // expected-error {{argument to '__builtin_lsx_vsrlri_d' must be a constant integer}}
  return res;
}

v16u8 vbitclri_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vbitclri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitclri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitclri_b(_1, var); // expected-error {{argument to '__builtin_lsx_vbitclri_b' must be a constant integer}}
  return res;
}

v8u16 vbitclri_h(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vbitclri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitclri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitclri_h(_1, var); // expected-error {{argument to '__builtin_lsx_vbitclri_h' must be a constant integer}}
  return res;
}

v4u32 vbitclri_w(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vbitclri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitclri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitclri_w(_1, var); // expected-error {{argument to '__builtin_lsx_vbitclri_w' must be a constant integer}}
  return res;
}

v2u64 vbitclri_d(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vbitclri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitclri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitclri_d(_1, var); // expected-error {{argument to '__builtin_lsx_vbitclri_d' must be a constant integer}}
  return res;
}

v16u8 vbitseti_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vbitseti_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitseti_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitseti_b(_1, var); // expected-error {{argument to '__builtin_lsx_vbitseti_b' must be a constant integer}}
  return res;
}

v8u16 vbitseti_h(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vbitseti_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitseti_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitseti_h(_1, var); // expected-error {{argument to '__builtin_lsx_vbitseti_h' must be a constant integer}}
  return res;
}

v4u32 vbitseti_w(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vbitseti_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitseti_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitseti_w(_1, var); // expected-error {{argument to '__builtin_lsx_vbitseti_w' must be a constant integer}}
  return res;
}

v2u64 vbitseti_d(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vbitseti_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitseti_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitseti_d(_1, var); // expected-error {{argument to '__builtin_lsx_vbitseti_d' must be a constant integer}}
  return res;
}

v16u8 vbitrevi_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vbitrevi_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitrevi_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vbitrevi_b(_1, var); // expected-error {{argument to '__builtin_lsx_vbitrevi_b' must be a constant integer}}
  return res;
}

v8u16 vbitrevi_h(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vbitrevi_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitrevi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vbitrevi_h(_1, var); // expected-error {{argument to '__builtin_lsx_vbitrevi_h' must be a constant integer}}
  return res;
}

v4u32 vbitrevi_w(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vbitrevi_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitrevi_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbitrevi_w(_1, var); // expected-error {{argument to '__builtin_lsx_vbitrevi_w' must be a constant integer}}
  return res;
}

v2u64 vbitrevi_d(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vbitrevi_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitrevi_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vbitrevi_d(_1, var); // expected-error {{argument to '__builtin_lsx_vbitrevi_d' must be a constant integer}}
  return res;
}

v16i8 vaddi_bu(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vaddi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vaddi_bu' must be a constant integer}}
  return res;
}

v8i16 vaddi_hu(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vaddi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vaddi_hu' must be a constant integer}}
  return res;
}

v4i32 vaddi_wu(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vaddi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vaddi_wu' must be a constant integer}}
  return res;
}

v2i64 vaddi_du(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vaddi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vaddi_du(_1, var); // expected-error {{argument to '__builtin_lsx_vaddi_du' must be a constant integer}}
  return res;
}

v16i8 vsubi_bu(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsubi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vsubi_bu' must be a constant integer}}
  return res;
}

v8i16 vsubi_hu(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsubi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vsubi_hu' must be a constant integer}}
  return res;
}

v4i32 vsubi_wu(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsubi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vsubi_wu' must be a constant integer}}
  return res;
}

v2i64 vsubi_du(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsubi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsubi_du(_1, var); // expected-error {{argument to '__builtin_lsx_vsubi_du' must be a constant integer}}
  return res;
}

v16i8 vmaxi_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vmaxi_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_b(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_b' must be a constant integer}}
  return res;
}

v8i16 vmaxi_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vmaxi_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_h(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_h' must be a constant integer}}
  return res;
}

v4i32 vmaxi_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vmaxi_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_w(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_w' must be a constant integer}}
  return res;
}

v2i64 vmaxi_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vmaxi_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmaxi_d(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_d' must be a constant integer}}
  return res;
}

v16u8 vmaxi_bu(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vmaxi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_bu' must be a constant integer}}
  return res;
}

v8u16 vmaxi_hu(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vmaxi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_hu' must be a constant integer}}
  return res;
}

v4u32 vmaxi_wu(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vmaxi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_wu' must be a constant integer}}
  return res;
}

v2u64 vmaxi_du(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vmaxi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmaxi_du(_1, var); // expected-error {{argument to '__builtin_lsx_vmaxi_du' must be a constant integer}}
  return res;
}

v16i8 vmini_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vmini_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_b(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_b' must be a constant integer}}
  return res;
}

v8i16 vmini_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vmini_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_h(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_h' must be a constant integer}}}
  return res;
}

v4i32 vmini_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vmini_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_w(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_w' must be a constant integer}}
  return res;
}

v2i64 vmini_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vmini_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vmini_d(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_d' must be a constant integer}}
  return res;
}

v16u8 vmini_bu(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vmini_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_bu' must be a constant integer}}
  return res;
}

v8u16 vmini_hu(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vmini_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_hu' must be a constant integer}}
  return res;
}

v4u32 vmini_wu(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vmini_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_wu' must be a constant integer}}
  return res;
}

v2u64 vmini_du(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vmini_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vmini_du(_1, var); // expected-error {{argument to '__builtin_lsx_vmini_du' must be a constant integer}}
  return res;
}

v16i8 vseqi_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vseqi_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_b(_1, var); // expected-error {{argument to '__builtin_lsx_vseqi_b' must be a constant integer}}
  return res;
}

v8i16 vseqi_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vseqi_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_h(_1, var); // expected-error {{argument to '__builtin_lsx_vseqi_h' must be a constant integer}}
  return res;
}

v4i32 vseqi_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vseqi_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_w(_1, var); // expected-error {{argument to '__builtin_lsx_vseqi_w' must be a constant integer}}
  return res;
}

v2i64 vseqi_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vseqi_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vseqi_d(_1, var); // expected-error {{argument to '__builtin_lsx_vseqi_d' must be a constant integer}}
  return res;
}

v16i8 vslti_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vslti_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_b(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_b' must be a constant integer}}
  return res;
}

v8i16 vslti_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vslti_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_h(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_h' must be a constant integer}}
  return res;
}

v4i32 vslti_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vslti_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_w(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_w' must be a constant integer}}
  return res;
}

v2i64 vslti_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vslti_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslti_d(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_d' must be a constant integer}}
  return res;
}

v16i8 vslti_bu(v16u8 _1, int var) {
  v16i8 res = __builtin_lsx_vslti_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_bu' must be a constant integer}}
  return res;
}

v8i16 vslti_hu(v8u16 _1, int var) {
  v8i16 res = __builtin_lsx_vslti_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_hu' must be a constant integer}}
  return res;
}

v4i32 vslti_wu(v4u32 _1, int var) {
  v4i32 res = __builtin_lsx_vslti_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_wu' must be a constant integer}}
  return res;
}

v2i64 vslti_du(v2u64 _1, int var) {
  v2i64 res = __builtin_lsx_vslti_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslti_du(_1, var); // expected-error {{argument to '__builtin_lsx_vslti_du' must be a constant integer}}
  return res;
}

v16i8 vslei_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vslei_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_b(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_b' must be a constant integer}}
  return res;
}

v8i16 vslei_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vslei_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_h(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_h' must be a constant integer}}
  return res;
}

v4i32 vslei_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vslei_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_w(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_w' must be a constant integer}}
  return res;
}

v2i64 vslei_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vslei_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lsx_vslei_d(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_d' must be a constant integer}}
  return res;
}

v16i8 vslei_bu(v16u8 _1, int var) {
  v16i8 res = __builtin_lsx_vslei_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_bu' must be a constant integer}}
  return res;
}

v8i16 vslei_hu(v8u16 _1, int var) {
  v8i16 res = __builtin_lsx_vslei_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_hu' must be a constant integer}}
  return res;
}

v4i32 vslei_wu(v4u32 _1, int var) {
  v4i32 res = __builtin_lsx_vslei_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_wu' must be a constant integer}}
  return res;
}

v2i64 vslei_du(v2u64 _1, int var) {
  v2i64 res = __builtin_lsx_vslei_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vslei_du(_1, var); // expected-error {{argument to '__builtin_lsx_vslei_du' must be a constant integer}}
  return res;
}

v16i8 vsat_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vsat_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsat_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsat_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_b' must be a constant integer}}
  return res;
}

v8i16 vsat_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vsat_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsat_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsat_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_h' must be a constant integer}}
  return res;
}

v4i32 vsat_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vsat_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsat_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsat_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_w' must be a constant integer}}
  return res;
}

v2i64 vsat_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vsat_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsat_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsat_d(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_d' must be a constant integer}}
  return res;
}

v16u8 vsat_bu(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vsat_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsat_bu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsat_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_bu' must be a constant integer}}
  return res;
}

v8u16 vsat_hu(v8u16 _1, int var) {
  v8u16 res = __builtin_lsx_vsat_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsat_hu(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsat_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_hu' must be a constant integer}}
  return res;
}

v4u32 vsat_wu(v4u32 _1, int var) {
  v4u32 res = __builtin_lsx_vsat_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsat_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsat_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_wu' must be a constant integer}}
  return res;
}

v2u64 vsat_du(v2u64 _1, int var) {
  v2u64 res = __builtin_lsx_vsat_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsat_du(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsat_du(_1, var); // expected-error {{argument to '__builtin_lsx_vsat_du' must be a constant integer}}
  return res;
}

v16i8 vreplvei_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vreplvei_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vreplvei_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vreplvei_b(_1, var); // expected-error {{argument to '__builtin_lsx_vreplvei_b' must be a constant integer}}
  return res;
}

v8i16 vreplvei_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vreplvei_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vreplvei_h(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vreplvei_h(_1, var); // expected-error {{argument to '__builtin_lsx_vreplvei_h' must be a constant integer}}
  return res;
}

v4i32 vreplvei_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vreplvei_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vreplvei_w(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vreplvei_w(_1, var); // expected-error {{argument to '__builtin_lsx_vreplvei_w' must be a constant integer}}
  return res;
}

v2i64 vreplvei_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vreplvei_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vreplvei_d(_1, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vreplvei_d(_1, var); // expected-error {{argument to '__builtin_lsx_vreplvei_d' must be a constant integer}}
  return res;
}

v16u8 vandi_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vandi_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vandi_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vandi_b(_1, var); // expected-error {{argument to '__builtin_lsx_vandi_b' must be a constant integer}}
  return res;
}

v16u8 vori_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vori_b(_1, var); // expected-error {{argument to '__builtin_lsx_vori_b' must be a constant integer}}
  return res;
}

v16u8 vnori_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vnori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vnori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vnori_b(_1, var); // expected-error {{argument to '__builtin_lsx_vnori_b' must be a constant integer}}
  return res;
}

v16u8 vxori_b(v16u8 _1, int var) {
  v16u8 res = __builtin_lsx_vxori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vxori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vxori_b(_1, var); // expected-error {{argument to '__builtin_lsx_vxori_b' must be a constant integer}}
  return res;
}

v16u8 vbitseli_b(v16u8 _1, v16u8 _2, int var) {
  v16u8 res = __builtin_lsx_vbitseli_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vbitseli_b(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vbitseli_b(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vbitseli_b' must be a constant integer}}
  return res;
}

v16i8 vshuf4i_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vshuf4i_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_b(_1, var); // expected-error {{argument to '__builtin_lsx_vshuf4i_b' must be a constant integer}}
  return res;
}

v8i16 vshuf4i_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vshuf4i_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_h(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_h(_1, var); // expected-error {{argument to '__builtin_lsx_vshuf4i_h' must be a constant integer}}
  return res;
}

v4i32 vshuf4i_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vshuf4i_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_w(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_w(_1, var); // expected-error {{argument to '__builtin_lsx_vshuf4i_w' must be a constant integer}}
  return res;
}

int vpickve2gr_b(v16i8 _1, int var) {
  int res = __builtin_lsx_vpickve2gr_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vpickve2gr_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vpickve2gr_b(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_b' must be a constant integer}}
  return res;
}

int vpickve2gr_h(v8i16 _1, int var) {
  int res = __builtin_lsx_vpickve2gr_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vpickve2gr_h(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vpickve2gr_h(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_h' must be a constant integer}}
  return res;
}

int vpickve2gr_w(v4i32 _1, int var) {
  int res = __builtin_lsx_vpickve2gr_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vpickve2gr_w(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vpickve2gr_w(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_w' must be a constant integer}}
  return res;
}

long vpickve2gr_d(v2i64 _1, int var) {
  long res = __builtin_lsx_vpickve2gr_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vpickve2gr_d(_1, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vpickve2gr_d(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_d' must be a constant integer}}
  return res;
}

unsigned int vpickve2gr_bu(v16i8 _1, int var) {
  unsigned int res = __builtin_lsx_vpickve2gr_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vpickve2gr_bu(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vpickve2gr_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_bu' must be a constant integer}}
  return res;
}

unsigned int vpickve2gr_hu(v8i16 _1, int var) {
  unsigned int res = __builtin_lsx_vpickve2gr_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vpickve2gr_hu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vpickve2gr_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_hu' must be a constant integer}}
  return res;
}

unsigned int vpickve2gr_wu(v4i32 _1, int var) {
  unsigned int res = __builtin_lsx_vpickve2gr_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vpickve2gr_wu(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vpickve2gr_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_wu' must be a constant integer}}
  return res;
}

unsigned long int vpickve2gr_du(v2i64 _1, int var) {
  unsigned long int res = __builtin_lsx_vpickve2gr_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vpickve2gr_du(_1, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vpickve2gr_du(_1, var); // expected-error {{argument to '__builtin_lsx_vpickve2gr_du' must be a constant integer}}
  return res;
}

v16i8 vinsgr2vr_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vinsgr2vr_b(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vinsgr2vr_b(_1, 1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vinsgr2vr_b(_1, 1, var); // expected-error {{argument to '__builtin_lsx_vinsgr2vr_b' must be a constant integer}}
  return res;
}

v8i16 vinsgr2vr_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vinsgr2vr_h(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vinsgr2vr_h(_1, 1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vinsgr2vr_h(_1, 1, var); // expected-error {{argument to '__builtin_lsx_vinsgr2vr_h' must be a constant integer}}
  return res;
}

v4i32 vinsgr2vr_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vinsgr2vr_w(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vinsgr2vr_w(_1, 1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lsx_vinsgr2vr_w(_1, 1, var); // expected-error {{argument to '__builtin_lsx_vinsgr2vr_w' must be a constant integer}}
  return res;
}

v2i64 vinsgr2vr_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vinsgr2vr_d(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vinsgr2vr_d(_1, 1, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res |= __builtin_lsx_vinsgr2vr_d(_1, 1, var); // expected-error {{argument to '__builtin_lsx_vinsgr2vr_d' must be a constant integer}}
  return res;
}

v8i16 vsllwil_h_b(v16i8 _1, int var) {
  v8i16 res = __builtin_lsx_vsllwil_h_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsllwil_h_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsllwil_h_b(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_h_b' must be a constant integer}}
  return res;
}

v4i32 vsllwil_w_h(v8i16 _1, int var) {
  v4i32 res = __builtin_lsx_vsllwil_w_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsllwil_w_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsllwil_w_h(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_w_h' must be a constant integer}}
  return res;
}

v2i64 vsllwil_d_w(v4i32 _1, int var) {
  v2i64 res = __builtin_lsx_vsllwil_d_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsllwil_d_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsllwil_d_w(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_d_w' must be a constant integer}}
  return res;
}

v8u16 vsllwil_hu_bu(v16u8 _1, int var) {
  v8u16 res = __builtin_lsx_vsllwil_hu_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsllwil_hu_bu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vsllwil_hu_bu(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_hu_bu' must be a constant integer}}
  return res;
}

v4u32 vsllwil_wu_hu(v8u16 _1, int var) {
  v4u32 res = __builtin_lsx_vsllwil_wu_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsllwil_wu_hu(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsllwil_wu_hu(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_wu_hu' must be a constant integer}}
  return res;
}

v2u64 vsllwil_du_wu(v4u32 _1, int var) {
  v2u64 res = __builtin_lsx_vsllwil_du_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsllwil_du_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsllwil_du_wu(_1, var); // expected-error {{argument to '__builtin_lsx_vsllwil_du_wu' must be a constant integer}}
  return res;
}

v16i8 vfrstpi_b(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vfrstpi_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vfrstpi_b(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vfrstpi_b(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vfrstpi_b' must be a constant integer}}
  return res;
}

v8i16 vfrstpi_h(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vfrstpi_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vfrstpi_h(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vfrstpi_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vfrstpi_h' must be a constant integer}}
  return res;
}

v2i64 vshuf4i_d(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vshuf4i_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_d(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vshuf4i_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vshuf4i_d' must be a constant integer}}
  return res;
}

v16i8 vbsrl_v(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vbsrl_v(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbsrl_v(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbsrl_v(_1, var); // expected-error {{argument to '__builtin_lsx_vbsrl_v' must be a constant integer}}
  return res;
}

v16i8 vbsll_v(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vbsll_v(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbsll_v(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vbsll_v(_1, var); // expected-error {{argument to '__builtin_lsx_vbsll_v' must be a constant integer}}
  return res;
}

v16i8 vextrins_b(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vextrins_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_b(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_b(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vextrins_b' must be a constant integer}}
  return res;
}

v8i16 vextrins_h(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vextrins_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_h(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vextrins_h' must be a constant integer}}
  return res;
}

v4i32 vextrins_w(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vextrins_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_w(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vextrins_w' must be a constant integer}}
  return res;
}

v2i64 vextrins_d(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vextrins_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_d(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vextrins_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vextrins_d' must be a constant integer}}
  return res;
}

void vstelm_b_idx(v16i8 _1, void *_2, int var) {
  __builtin_lsx_vstelm_b(_1, _2, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  __builtin_lsx_vstelm_b(_1, _2, 1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  __builtin_lsx_vstelm_b(_1, _2, 1, var); // expected-error {{argument to '__builtin_lsx_vstelm_b' must be a constant integer}}
}

void vstelm_h_idx(v8i16 _1, void *_2, int var) {
  __builtin_lsx_vstelm_h(_1, _2, 2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  __builtin_lsx_vstelm_h(_1, _2, 2, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  __builtin_lsx_vstelm_h(_1, _2, 2, var); // expected-error {{argument to '__builtin_lsx_vstelm_h' must be a constant integer}}
}

void vstelm_w_idx(v4i32 _1, void *_2, int var) {
  __builtin_lsx_vstelm_w(_1, _2, 4, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_lsx_vstelm_w(_1, _2, 4, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_lsx_vstelm_w(_1, _2, 4, var); // expected-error {{argument to '__builtin_lsx_vstelm_w' must be a constant integer}}
}

void vstelm_d_idx(v2i64 _1, void *_2, int var) {
  __builtin_lsx_vstelm_d(_1, _2, 8, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  __builtin_lsx_vstelm_d(_1, _2, 8, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  __builtin_lsx_vstelm_d(_1, _2, 8, var); // expected-error {{argument to '__builtin_lsx_vstelm_d' must be a constant integer}}
}

void vstelm_b(v16i8 _1, void *_2, int var) {
  __builtin_lsx_vstelm_b(_1, _2, -129, 1); // expected-error {{argument value -129 is outside the valid range [-128, 127]}}
  __builtin_lsx_vstelm_b(_1, _2, 128, 1); // expected-error {{argument value 128 is outside the valid range [-128, 127]}}
  __builtin_lsx_vstelm_b(_1, _2, var, 1); // expected-error {{argument to '__builtin_lsx_vstelm_b' must be a constant integer}}
}

void vstelm_h(v8i16 _1, void *_2, int var) {
  __builtin_lsx_vstelm_h(_1, _2, -258, 1); // expected-error {{argument value -258 is outside the valid range [-256, 254]}}
  __builtin_lsx_vstelm_h(_1, _2, 256, 1); // expected-error {{argument value 256 is outside the valid range [-256, 254]}}
  __builtin_lsx_vstelm_h(_1, _2, var, 1); // expected-error {{argument to '__builtin_lsx_vstelm_h' must be a constant integer}}
}

void vstelm_w(v4i32 _1, void *_2, int var) {
  __builtin_lsx_vstelm_w(_1, _2, -516, 1); // expected-error {{argument value -516 is outside the valid range [-512, 508]}}
  __builtin_lsx_vstelm_w(_1, _2, 512, 1); // expected-error {{argument value 512 is outside the valid range [-512, 508]}}
  __builtin_lsx_vstelm_w(_1, _2, var, 1); // expected-error {{argument to '__builtin_lsx_vstelm_w' must be a constant integer}}
}

void vstelm_d(v2i64 _1, void *_2, int var) {
  __builtin_lsx_vstelm_d(_1, _2, -1032, 1); // expected-error {{argument value -1032 is outside the valid range [-1024, 1016]}}
  __builtin_lsx_vstelm_d(_1, _2, 1024, 1); // expected-error {{argument value 1024 is outside the valid range [-1024, 1016]}}
  __builtin_lsx_vstelm_d(_1, _2, var, 1); // expected-error {{argument to '__builtin_lsx_vstelm_d' must be a constant integer}}
}

v16i8 vldrepl_b(void *_1, int var) {
  v16i8 res = __builtin_lsx_vldrepl_b(_1, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lsx_vldrepl_b(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lsx_vldrepl_b(_1, var); // expected-error {{argument to '__builtin_lsx_vldrepl_b' must be a constant integer}}
  return res;
}

v8i16 vldrepl_h(void *_1, int var) {
  v8i16 res = __builtin_lsx_vldrepl_h(_1, -2050); // expected-error {{argument value -2050 is outside the valid range [-2048, 2046]}}
  res |= __builtin_lsx_vldrepl_h(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2046]}}
  res |= __builtin_lsx_vldrepl_h(_1, var); // expected-error {{argument to '__builtin_lsx_vldrepl_h' must be a constant integer}}
  return res;
}

v4i32 vldrepl_w(void *_1, int var) {
  v4i32 res = __builtin_lsx_vldrepl_w(_1, -2052); // expected-error {{argument value -2052 is outside the valid range [-2048, 2044]}}
  res |= __builtin_lsx_vldrepl_w(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2044]}}
  res |= __builtin_lsx_vldrepl_w(_1, var); // expected-error {{argument to '__builtin_lsx_vldrepl_w' must be a constant integer}}
  return res;
}

v2i64 vldrepl_d(void *_1, int var) {
  v2i64 res = __builtin_lsx_vldrepl_d(_1, -2056); // expected-error {{argument value -2056 is outside the valid range [-2048, 2040]}}
  res |= __builtin_lsx_vldrepl_d(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2040]}}
  res |= __builtin_lsx_vldrepl_d(_1, var); // expected-error {{argument to '__builtin_lsx_vldrepl_d' must be a constant integer}}
  return res;
}

v16i8 vrotri_b(v16i8 _1, int var) {
  v16i8 res = __builtin_lsx_vrotri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vrotri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lsx_vrotri_b(_1, var); // expected-error {{argument to '__builtin_lsx_vrotri_b' must be a constant integer}}
  return res;
}

v8i16 vrotri_h(v8i16 _1, int var) {
  v8i16 res = __builtin_lsx_vrotri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vrotri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vrotri_h(_1, var); // expected-error {{argument to '__builtin_lsx_vrotri_h' must be a constant integer}}
  return res;
}

v4i32 vrotri_w(v4i32 _1, int var) {
  v4i32 res = __builtin_lsx_vrotri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vrotri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vrotri_w(_1, var); // expected-error {{argument to '__builtin_lsx_vrotri_w' must be a constant integer}}
  return res;
}

v2i64 vrotri_d(v2i64 _1, int var) {
  v2i64 res = __builtin_lsx_vrotri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vrotri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vrotri_d(_1, var); // expected-error {{argument to '__builtin_lsx_vrotri_d' must be a constant integer}}
  return res;
}

v16i8 vsrlni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vsrlni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlni_b_h' must be a constant integer}}
  return res;
}

v8i16 vsrlni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vsrlni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlni_h_w' must be a constant integer}}
  return res;
}

v4i32 vsrlni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vsrlni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlni_w_d' must be a constant integer}}
  return res;
}

v2i64 vsrlni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vsrlni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrlni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrlni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlni_d_q' must be a constant integer}}
  return res;
}

v16i8 vsrlrni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vsrlrni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlrni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrlrni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlrni_b_h' must be a constant integer}}
  return res;
}

v8i16 vsrlrni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vsrlrni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlrni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrlrni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlrni_h_w' must be a constant integer}}
  return res;
}

v4i32 vsrlrni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vsrlrni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlrni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrlrni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlrni_w_d' must be a constant integer}}
  return res;
}

v2i64 vsrlrni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vsrlrni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrlrni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrlrni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrlrni_d_q' must be a constant integer}}
  return res;
}

v16i8 vssrlni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vssrlni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_b_h' must be a constant integer}}
  return res;
}

v8i16 vssrlni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vssrlni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_h_w' must be a constant integer}}
  return res;
}

v4i32 vssrlni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vssrlni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_w_d' must be a constant integer}}
  return res;
}

v2i64 vssrlni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vssrlni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_d_q' must be a constant integer}}
  return res;
}

v16u8 vssrlni_bu_h(v16u8 _1, v16i8 _2, int var) {
  v16u8 res = __builtin_lsx_vssrlni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_bu_h' must be a constant integer}}
  return res;
}

v8u16 vssrlni_hu_w(v8u16 _1, v8i16 _2, int var) {
  v8u16 res = __builtin_lsx_vssrlni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_hu_w' must be a constant integer}}
  return res;
}

v4u32 vssrlni_wu_d(v4u32 _1, v4i32 _2, int var) {
  v4u32 res = __builtin_lsx_vssrlni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_wu_d' must be a constant integer}}
  return res;
}

v2u64 vssrlni_du_q(v2u64 _1, v2i64 _2, int var) {
  v2u64 res = __builtin_lsx_vssrlni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlni_du_q' must be a constant integer}}
  return res;
}

v16i8 vssrlrni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vssrlrni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlrni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlrni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_b_h' must be a constant integer}}
  return res;
}

v8i16 vssrlrni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vssrlrni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlrni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlrni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_h_w' must be a constant integer}}
  return res;
}

v4i32 vssrlrni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vssrlrni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlrni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlrni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_w_d' must be a constant integer}}
  return res;
}

v2i64 vssrlrni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vssrlrni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlrni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlrni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_d_q' must be a constant integer}}
  return res;
}

v16u8 vssrlrni_bu_h(v16u8 _1, v16i8 _2, int var) {
  v16u8 res = __builtin_lsx_vssrlrni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlrni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrlrni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_bu_h' must be a constant integer}}
  return res;
}

v8u16 vssrlrni_hu_w(v8u16 _1, v8i16 _2, int var) {
  v8u16 res = __builtin_lsx_vssrlrni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlrni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrlrni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_hu_w' must be a constant integer}}
  return res;
}

v4u32 vssrlrni_wu_d(v4u32 _1, v4i32 _2, int var) {
  v4u32 res = __builtin_lsx_vssrlrni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlrni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrlrni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_wu_d' must be a constant integer}}
  return res;
}

v2u64 vssrlrni_du_q(v2u64 _1, v2i64 _2, int var) {
  v2u64 res = __builtin_lsx_vssrlrni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlrni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrlrni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrlrni_du_q' must be a constant integer}}
  return res;
}

v16i8 vsrani_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vsrani_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrani_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrani_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrani_b_h' must be a constant integer}}
  return res;
}

v8i16 vsrani_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vsrani_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrani_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrani_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrani_h_w' must be a constant integer}}
  return res;
}

v4i32 vsrani_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vsrani_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrani_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrani_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrani_w_d' must be a constant integer}}
  return res;
}

v2i64 vsrani_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vsrani_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrani_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrani_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrani_d_q' must be a constant integer}}
  return res;
}

v16i8 vsrarni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vsrarni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrarni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vsrarni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrarni_b_h' must be a constant integer}}
  return res;
}

v8i16 vsrarni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vsrarni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrarni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vsrarni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrarni_h_w' must be a constant integer}}
  return res;
}

v4i32 vsrarni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vsrarni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrarni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vsrarni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrarni_w_d' must be a constant integer}}
  return res;
}

v2i64 vsrarni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vsrarni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrarni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vsrarni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vsrarni_d_q' must be a constant integer}}
  return res;
}

v16i8 vssrani_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vssrani_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrani_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrani_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_b_h' must be a constant integer}}
  return res;
}

v8i16 vssrani_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vssrani_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrani_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrani_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_h_w' must be a constant integer}}
  return res;
}

v4i32 vssrani_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vssrani_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrani_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrani_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_w_d' must be a constant integer}}
  return res;
}

v2i64 vssrani_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vssrani_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrani_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrani_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_d_q' must be a constant integer}}
  return res;
}

v16u8 vssrani_bu_h(v16u8 _1, v16i8 _2, int var) {
  v16u8 res = __builtin_lsx_vssrani_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrani_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrani_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_bu_h' must be a constant integer}}
  return res;
}

v8u16 vssrani_hu_w(v8u16 _1, v8i16 _2, int var) {
  v8u16 res = __builtin_lsx_vssrani_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrani_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrani_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_hu_w' must be a constant integer}}
  return res;
}

v4u32 vssrani_wu_d(v4u32 _1, v4i32 _2, int var) {
  v4u32 res = __builtin_lsx_vssrani_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrani_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrani_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_wu_d' must be a constant integer}}
  return res;
}

v2u64 vssrani_du_q(v2u64 _1, v2i64 _2, int var) {
  v2u64 res = __builtin_lsx_vssrani_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrani_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrani_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrani_du_q' must be a constant integer}}
  return res;
}

v16i8 vssrarni_b_h(v16i8 _1, v16i8 _2, int var) {
  v16i8 res = __builtin_lsx_vssrarni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrarni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrarni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_b_h' must be a constant integer}}
  return res;
}

v8i16 vssrarni_h_w(v8i16 _1, v8i16 _2, int var) {
  v8i16 res = __builtin_lsx_vssrarni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrarni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrarni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_h_w' must be a constant integer}}
  return res;
}

v4i32 vssrarni_w_d(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vssrarni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrarni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrarni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_w_d' must be a constant integer}}
  return res;
}

v2i64 vssrarni_d_q(v2i64 _1, v2i64 _2, int var) {
  v2i64 res = __builtin_lsx_vssrarni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrarni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrarni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_d_q' must be a constant integer}}
  return res;
}

v16u8 vssrarni_bu_h(v16u8 _1, v16i8 _2, int var) {
  v16u8 res = __builtin_lsx_vssrarni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrarni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lsx_vssrarni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_bu_h' must be a constant integer}}
  return res;
}

v8u16 vssrarni_hu_w(v8u16 _1, v8i16 _2, int var) {
  v8u16 res = __builtin_lsx_vssrarni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrarni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lsx_vssrarni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_hu_w' must be a constant integer}}
  return res;
}

v4u32 vssrarni_wu_d(v4u32 _1, v4i32 _2, int var) {
  v4u32 res = __builtin_lsx_vssrarni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrarni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lsx_vssrarni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_wu_d' must be a constant integer}}
  return res;
}

v2u64 vssrarni_du_q(v2u64 _1, v2i64 _2, int var) {
  v2u64 res = __builtin_lsx_vssrarni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrarni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lsx_vssrarni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vssrarni_du_q' must be a constant integer}}
  return res;
}

v4i32 vpermi_w(v4i32 _1, v4i32 _2, int var) {
  v4i32 res = __builtin_lsx_vpermi_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vpermi_w(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lsx_vpermi_w(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vpermi_w' must be a constant integer}}
  return res;
}

v16i8 vld(void *_1, int var) {
  v16i8 res = __builtin_lsx_vld(_1, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lsx_vld(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lsx_vld(_1, var); // expected-error {{argument to '__builtin_lsx_vld' must be a constant integer}}
  return res;
}

void vst(v16i8 _1, void *_2, int var) {
  __builtin_lsx_vst(_1, _2, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  __builtin_lsx_vst(_1, _2, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  __builtin_lsx_vst(_1, _2, var); // expected-error {{argument to '__builtin_lsx_vst' must be a constant integer}}
}

v2i64 vldi(int var) {
  v2i64 res = __builtin_lsx_vldi(-4097); // expected-error {{argument value -4097 is outside the valid range [-4096, 4095]}}
  res |= __builtin_lsx_vldi(4096); // expected-error {{argument value 4096 is outside the valid range [-4096, 4095]}}
  res |= __builtin_lsx_vldi(var); // expected-error {{argument to '__builtin_lsx_vldi' must be a constant integer}}
  return res;
}

v16i8 vrepli_b(int var) {
  v16i8 res = __builtin_lsx_vrepli_b(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_b(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_b(var); // expected-error {{argument to '__builtin_lsx_vrepli_b' must be a constant integer}}
  return res;
}

v2i64 vrepli_d(int var) {
  v2i64 res = __builtin_lsx_vrepli_d(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_d(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_d(var); // expected-error {{argument to '__builtin_lsx_vrepli_d' must be a constant integer}}
  return res;
}

v8i16 vrepli_h(int var) {
  v8i16 res = __builtin_lsx_vrepli_h(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_h(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_h(var); // expected-error {{argument to '__builtin_lsx_vrepli_h' must be a constant integer}}
  return res;
}

v4i32 vrepli_w(int var) {
  v4i32 res = __builtin_lsx_vrepli_w(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_w(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lsx_vrepli_w(var); // expected-error {{argument to '__builtin_lsx_vrepli_w' must be a constant integer}}
  return res;
}
