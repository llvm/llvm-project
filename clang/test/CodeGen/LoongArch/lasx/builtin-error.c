// RUN: %clang_cc1 -triple loongarch64 -target-feature +lasx -verify %s

typedef signed char v32i8 __attribute__((vector_size(32), aligned(32)));
typedef signed char v32i8_b __attribute__((vector_size(32), aligned(1)));
typedef unsigned char v32u8 __attribute__((vector_size(32), aligned(32)));
typedef unsigned char v32u8_b __attribute__((vector_size(32), aligned(1)));
typedef short v16i16 __attribute__((vector_size(32), aligned(32)));
typedef short v16i16_h __attribute__((vector_size(32), aligned(2)));
typedef unsigned short v16u16 __attribute__((vector_size(32), aligned(32)));
typedef unsigned short v16u16_h __attribute__((vector_size(32), aligned(2)));
typedef int v8i32 __attribute__((vector_size(32), aligned(32)));
typedef int v8i32_w __attribute__((vector_size(32), aligned(4)));
typedef unsigned int v8u32 __attribute__((vector_size(32), aligned(32)));
typedef unsigned int v8u32_w __attribute__((vector_size(32), aligned(4)));
typedef long long v4i64 __attribute__((vector_size(32), aligned(32)));
typedef long long v4i64_d __attribute__((vector_size(32), aligned(8)));
typedef unsigned long long v4u64 __attribute__((vector_size(32), aligned(32)));
typedef unsigned long long v4u64_d __attribute__((vector_size(32), aligned(8)));
typedef float v8f32 __attribute__((vector_size(32), aligned(32)));
typedef float v8f32_w __attribute__((vector_size(32), aligned(4)));
typedef double v4f64 __attribute__((vector_size(32), aligned(32)));
typedef double v4f64_d __attribute__((vector_size(32), aligned(8)));

v32i8 xvslli_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvslli_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvslli_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvslli_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvslli_b' must be a constant integer}}
  return res;
}

v16i16 xvslli_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvslli_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvslli_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvslli_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvslli_h' must be a constant integer}}
  return res;
}

v8i32 xvslli_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvslli_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslli_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslli_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvslli_w' must be a constant integer}}
  return res;
}

v4i64 xvslli_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvslli_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvslli_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvslli_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvslli_d' must be a constant integer}}
  return res;
}

v32i8 xvsrai_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsrai_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrai_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrai_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrai_b' must be a constant integer}}
  return res;
}

v16i16 xvsrai_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsrai_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrai_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrai_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrai_h' must be a constant integer}}
  return res;
}

v8i32 xvsrai_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsrai_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrai_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrai_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrai_w' must be a constant integer}}
  return res;
}

v4i64 xvsrai_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsrai_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrai_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrai_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrai_d' must be a constant integer}}
  return res;
}

v32i8 xvsrari_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsrari_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrari_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrari_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrari_b' must be a constant integer}}
  return res;
}

v16i16 xvsrari_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsrari_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrari_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrari_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrari_h' must be a constant integer}}
  return res;
}

v8i32 xvsrari_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsrari_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrari_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrari_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrari_w' must be a constant integer}}
  return res;
}

v4i64 xvsrari_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsrari_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrari_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrari_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrari_d' must be a constant integer}}
  return res;
}

v32i8 xvsrli_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsrli_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrli_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrli_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrli_b' must be a constant integer}}
  return res;
}

v16i16 xvsrli_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsrli_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrli_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrli_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrli_h' must be a constant integer}}
  return res;
}

v8i32 xvsrli_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsrli_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrli_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrli_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrli_w' must be a constant integer}}
  return res;
}

v4i64 xvsrli_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsrli_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrli_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrli_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrli_d' must be a constant integer}}
  return res;
}

v32i8 xvsrlri_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsrlri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrlri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsrlri_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrlri_b' must be a constant integer}}
  return res;
}

v16i16 xvsrlri_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsrlri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlri_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrlri_h' must be a constant integer}}
  return res;
}

v8i32 xvsrlri_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsrlri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlri_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrlri_w' must be a constant integer}}
  return res;
}

v4i64 xvsrlri_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsrlri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlri_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvsrlri_d' must be a constant integer}}
  return res;
}

v32u8 xvbitclri_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvbitclri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitclri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitclri_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitclri_b' must be a constant integer}}
  return res;
}

v16u16 xvbitclri_h(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvbitclri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitclri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitclri_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitclri_h' must be a constant integer}}
  return res;
}

v8u32 xvbitclri_w(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvbitclri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitclri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitclri_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitclri_w' must be a constant integer}}
  return res;
}

v4u64 xvbitclri_d(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvbitclri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitclri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitclri_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitclri_d' must be a constant integer}}
  return res;
}

v32u8 xvbitseti_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvbitseti_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitseti_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitseti_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitseti_b' must be a constant integer}}
  return res;
}

v16u16 xvbitseti_h(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvbitseti_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitseti_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitseti_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitseti_h' must be a constant integer}}
  return res;
}

v8u32 xvbitseti_w(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvbitseti_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitseti_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitseti_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitseti_w' must be a constant integer}}
  return res;
}

v4u64 xvbitseti_d(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvbitseti_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitseti_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitseti_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitseti_d' must be a constant integer}}
  return res;
}

v32u8 xvbitrevi_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvbitrevi_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitrevi_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvbitrevi_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitrevi_b' must be a constant integer}}
  return res;
}

v16u16 xvbitrevi_h(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvbitrevi_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitrevi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvbitrevi_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitrevi_h' must be a constant integer}}
  return res;
}

v8u32 xvbitrevi_w(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvbitrevi_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitrevi_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbitrevi_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitrevi_w' must be a constant integer}}
  return res;
}

v4u64 xvbitrevi_d(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvbitrevi_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitrevi_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvbitrevi_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvbitrevi_d' must be a constant integer}}
  return res;
}

v32i8 xvaddi_bu(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvaddi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvaddi_bu' must be a constant integer}}
  return res;
}

v16i16 xvaddi_hu(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvaddi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvaddi_hu' must be a constant integer}}
  return res;
}

v8i32 xvaddi_wu(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvaddi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvaddi_wu' must be a constant integer}}
  return res;
}

v4i64 xvaddi_du(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvaddi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvaddi_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvaddi_du' must be a constant integer}}
  return res;
}

v32i8 xvsubi_bu(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsubi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsubi_bu' must be a constant integer}}
  return res;
}

v16i16 xvsubi_hu(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsubi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsubi_hu' must be a constant integer}}
  return res;
}

v8i32 xvsubi_wu(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsubi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsubi_wu' must be a constant integer}}
  return res;
}

v4i64 xvsubi_du(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsubi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsubi_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvsubi_du' must be a constant integer}}
  return res;
}

v32i8 xvmaxi_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvmaxi_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_b' must be a constant integer}}
  return res;
}

v16i16 xvmaxi_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvmaxi_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_h' must be a constant integer}}
  return res;
}

v8i32 xvmaxi_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvmaxi_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_w' must be a constant integer}}
  return res;
}

v4i64 xvmaxi_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvmaxi_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmaxi_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_d' must be a constant integer}}
  return res;
}

v32u8 xvmaxi_bu(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvmaxi_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_bu' must be a constant integer}}
  return res;
}

v16u16 xvmaxi_hu(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvmaxi_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_hu' must be a constant integer}}
  return res;
}

v8u32 xvmaxi_wu(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvmaxi_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_wu' must be a constant integer}}
  return res;
}

v4u64 xvmaxi_du(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvmaxi_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmaxi_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvmaxi_du' must be a constant integer}}
  return res;
}

v32i8 xvmini_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvmini_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_b' must be a constant integer}}
  return res;
}

v16i16 xvmini_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvmini_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_h' must be a constant integer}}}
  return res;
}

v8i32 xvmini_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvmini_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_w' must be a constant integer}}
  return res;
}

v4i64 xvmini_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvmini_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvmini_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_d' must be a constant integer}}
  return res;
}

v32u8 xvmini_bu(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvmini_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_bu' must be a constant integer}}
  return res;
}

v16u16 xvmini_hu(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvmini_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_hu' must be a constant integer}}
  return res;
}

v8u32 xvmini_wu(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvmini_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_wu' must be a constant integer}}
  return res;
}

v4u64 xvmini_du(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvmini_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvmini_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvmini_du' must be a constant integer}}
  return res;
}

v32i8 xvseqi_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvseqi_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvseqi_b' must be a constant integer}}
  return res;
}

v16i16 xvseqi_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvseqi_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvseqi_h' must be a constant integer}}
  return res;
}

v8i32 xvseqi_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvseqi_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvseqi_w' must be a constant integer}}
  return res;
}

v4i64 xvseqi_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvseqi_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvseqi_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvseqi_d' must be a constant integer}}
  return res;
}

v32i8 xvslti_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvslti_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_b' must be a constant integer}}
  return res;
}

v16i16 xvslti_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvslti_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_h' must be a constant integer}}
  return res;
}

v8i32 xvslti_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvslti_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_w' must be a constant integer}}
  return res;
}

v4i64 xvslti_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvslti_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslti_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_d' must be a constant integer}}
  return res;
}

v32i8 xvslti_bu(v32u8 _1, int var) {
  v32i8 res = __builtin_lasx_xvslti_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_bu' must be a constant integer}}
  return res;
}

v16i16 xvslti_hu(v16u16 _1, int var) {
  v16i16 res = __builtin_lasx_xvslti_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_hu' must be a constant integer}}
  return res;
}

v8i32 xvslti_wu(v8u32 _1, int var) {
  v8i32 res = __builtin_lasx_xvslti_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_wu' must be a constant integer}}
  return res;
}

v4i64 xvslti_du(v4u64 _1, int var) {
  v4i64 res = __builtin_lasx_xvslti_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslti_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvslti_du' must be a constant integer}}
  return res;
}

v32i8 xvslei_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvslei_b(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_b' must be a constant integer}}
  return res;
}

v16i16 xvslei_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvslei_h(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_h' must be a constant integer}}
  return res;
}

v8i32 xvslei_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvslei_w(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_w(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_w' must be a constant integer}}
  return res;
}

v4i64 xvslei_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvslei_d(_1, -17); // expected-error {{argument value -17 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_d(_1, 16); // expected-error {{argument value 16 is outside the valid range [-16, 15]}}
  res |= __builtin_lasx_xvslei_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_d' must be a constant integer}}
  return res;
}

v32i8 xvslei_bu(v32u8 _1, int var) {
  v32i8 res = __builtin_lasx_xvslei_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_bu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_bu' must be a constant integer}}
  return res;
}

v16i16 xvslei_hu(v16u16 _1, int var) {
  v16i16 res = __builtin_lasx_xvslei_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_hu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_hu' must be a constant integer}}
  return res;
}

v8i32 xvslei_wu(v8u32 _1, int var) {
  v8i32 res = __builtin_lasx_xvslei_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_wu' must be a constant integer}}
  return res;
}

v4i64 xvslei_du(v4u64 _1, int var) {
  v4i64 res = __builtin_lasx_xvslei_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_du(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvslei_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvslei_du' must be a constant integer}}
  return res;
}

v32i8 xvsat_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvsat_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsat_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsat_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_b' must be a constant integer}}
  return res;
}

v16i16 xvsat_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvsat_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsat_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsat_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_h' must be a constant integer}}
  return res;
}

v8i32 xvsat_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvsat_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsat_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsat_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_w' must be a constant integer}}
  return res;
}

v4i64 xvsat_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvsat_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsat_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsat_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_d' must be a constant integer}}
  return res;
}

v32u8 xvsat_bu(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvsat_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsat_bu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsat_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_bu' must be a constant integer}}
  return res;
}

v16u16 xvsat_hu(v16u16 _1, int var) {
  v16u16 res = __builtin_lasx_xvsat_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsat_hu(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsat_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_hu' must be a constant integer}}
  return res;
}

v8u32 xvsat_wu(v8u32 _1, int var) {
  v8u32 res = __builtin_lasx_xvsat_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsat_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsat_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_wu' must be a constant integer}}
  return res;
}

v4u64 xvsat_du(v4u64 _1, int var) {
  v4u64 res = __builtin_lasx_xvsat_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsat_du(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsat_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvsat_du' must be a constant integer}}
  return res;
}

v32i8 xvrepl128vei_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvrepl128vei_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvrepl128vei_b(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvrepl128vei_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvrepl128vei_b' must be a constant integer}}
  return res;
}

v16i16 xvrepl128vei_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvrepl128vei_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvrepl128vei_h(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvrepl128vei_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvrepl128vei_h' must be a constant integer}}
  return res;
}

v8i32 xvrepl128vei_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvrepl128vei_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvrepl128vei_w(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvrepl128vei_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvrepl128vei_w' must be a constant integer}}
  return res;
}

v4i64 xvrepl128vei_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvrepl128vei_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 1]}}
  res |= __builtin_lasx_xvrepl128vei_d(_1, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  res |= __builtin_lasx_xvrepl128vei_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvrepl128vei_d' must be a constant integer}}
  return res;
}

v32u8 xvandi_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvandi_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvandi_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvandi_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvandi_b' must be a constant integer}}
  return res;
}

v32u8 xvori_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvori_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvori_b' must be a constant integer}}
  return res;
}

v32u8 xvnori_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvnori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvnori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvnori_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvnori_b' must be a constant integer}}
  return res;
}

v32u8 xvxori_b(v32u8 _1, int var) {
  v32u8 res = __builtin_lasx_xvxori_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvxori_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvxori_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvxori_b' must be a constant integer}}
  return res;
}

v32u8 xvbitseli_b(v32u8 _1, v32u8 _2, int var) {
  v32u8 res = __builtin_lasx_xvbitseli_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvbitseli_b(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvbitseli_b(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvbitseli_b' must be a constant integer}}
  return res;
}

v32i8 xvshuf4i_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvshuf4i_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_b(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvshuf4i_b' must be a constant integer}}
  return res;
}

v16i16 xvshuf4i_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvshuf4i_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_h(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvshuf4i_h' must be a constant integer}}
  return res;
}

v8i32 xvshuf4i_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvshuf4i_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_w(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvshuf4i_w' must be a constant integer}}
  return res;
}

v4i64 xvshuf4i_d(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvshuf4i_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_d(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvshuf4i_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvshuf4i_d' must be a constant integer}}
  return res;
}

v8i32 xvpermi_w(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvpermi_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_w(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvpermi_w' must be a constant integer}}
  return res;
}

v4i64 xvpermi_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvpermi_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_d(_1, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvpermi_d' must be a constant integer}}
  return res;
}

v32i8 xvpermi_q(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvpermi_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_q(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvpermi_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvpermi_q' must be a constant integer}}
  return res;
}

v16i16 xvsllwil_h_b(v32i8 _1, int var) {
  v16i16 res = __builtin_lasx_xvsllwil_h_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsllwil_h_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsllwil_h_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_h_b' must be a constant integer}}
  return res;
}

v8i32 xvsllwil_w_h(v16i16 _1, int var) {
  v8i32 res = __builtin_lasx_xvsllwil_w_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsllwil_w_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsllwil_w_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_w_h' must be a constant integer}}
  return res;
}

v4i64 xvsllwil_d_w(v8i32 _1, int var) {
  v4i64 res = __builtin_lasx_xvsllwil_d_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsllwil_d_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsllwil_d_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_d_w' must be a constant integer}}
  return res;
}

v16u16 xvsllwil_hu_bu(v32u8 _1, int var) {
  v16u16 res = __builtin_lasx_xvsllwil_hu_bu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsllwil_hu_bu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvsllwil_hu_bu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_hu_bu' must be a constant integer}}
  return res;
}

v8u32 xvsllwil_wu_hu(v16u16 _1, int var) {
  v8u32 res = __builtin_lasx_xvsllwil_wu_hu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsllwil_wu_hu(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsllwil_wu_hu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_wu_hu' must be a constant integer}}
  return res;
}

v4u64 xvsllwil_du_wu(v8u32 _1, int var) {
  v4u64 res = __builtin_lasx_xvsllwil_du_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsllwil_du_wu(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsllwil_du_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvsllwil_du_wu' must be a constant integer}}
  return res;
}

v32i8 xvfrstpi_b(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvfrstpi_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvfrstpi_b(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvfrstpi_b(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvfrstpi_b' must be a constant integer}}
  return res;
}

v16i16 xvfrstpi_h(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvfrstpi_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvfrstpi_h(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvfrstpi_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvfrstpi_h' must be a constant integer}}
  return res;
}

v32i8 xvbsrl_v(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvbsrl_v(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbsrl_v(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbsrl_v(_1, var); // expected-error {{argument to '__builtin_lasx_xvbsrl_v' must be a constant integer}}
  return res;
}

v32i8 xvbsll_v(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvbsll_v(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbsll_v(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvbsll_v(_1, var); // expected-error {{argument to '__builtin_lasx_xvbsll_v' must be a constant integer}}
  return res;
}

v32i8 xvextrins_b(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvextrins_b(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_b(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_b(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvextrins_b' must be a constant integer}}
  return res;
}

v16i16 xvextrins_h(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvextrins_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_h(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvextrins_h' must be a constant integer}}
  return res;
}

v8i32 xvextrins_w(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvextrins_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_w(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvextrins_w' must be a constant integer}}
  return res;
}

v4i64 xvextrins_d(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvextrins_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_d(_1, _2, 256); // expected-error {{argument value 256 is outside the valid range [0, 255]}}
  res |= __builtin_lasx_xvextrins_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvextrins_d' must be a constant integer}}
  return res;
}

v32i8 xvld(void *_1, int var) {
  v32i8 res = __builtin_lasx_xvld(_1, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lasx_xvld(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lasx_xvld(_1, var); // expected-error {{argument to '__builtin_lasx_xvld' must be a constant integer}}
  return res;
}

void xvst(v32i8 _1, void *_2, int var) {
  __builtin_lasx_xvst(_1, _2, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  __builtin_lasx_xvst(_1, _2, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  __builtin_lasx_xvst(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvst' must be a constant integer}}
}

void xvstelm_b(v32i8 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_b(_1, _2, -129, 1); // expected-error {{argument value -129 is outside the valid range [-128, 127]}}
  __builtin_lasx_xvstelm_b(_1, _2, 128, 1); // expected-error {{argument value 128 is outside the valid range [-128, 127]}}
  __builtin_lasx_xvstelm_b(_1, _2, var, 1); // expected-error {{argument to '__builtin_lasx_xvstelm_b' must be a constant integer}}
}

void xvstelm_h(v16i16 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_h(_1, _2, -258, 1); // expected-error {{argument value -258 is outside the valid range [-256, 254]}}
  __builtin_lasx_xvstelm_h(_1, _2, 256, 1); // expected-error {{argument value 256 is outside the valid range [-256, 254]}}
  __builtin_lasx_xvstelm_h(_1, _2, var, 1); // expected-error {{argument to '__builtin_lasx_xvstelm_h' must be a constant integer}}
}

void xvstelm_w(v8i32 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_w(_1, _2, -516, 1); // expected-error {{argument value -516 is outside the valid range [-512, 508]}}
  __builtin_lasx_xvstelm_w(_1, _2, 512, 1); // expected-error {{argument value 512 is outside the valid range [-512, 508]}}
  __builtin_lasx_xvstelm_w(_1, _2, var, 1); // expected-error {{argument to '__builtin_lasx_xvstelm_w' must be a constant integer}}
}

void xvstelm_d(v4i64 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_d(_1, _2, -1032, 1); // expected-error {{argument value -1032 is outside the valid range [-1024, 1016]}}
  __builtin_lasx_xvstelm_d(_1, _2, 1024, 1); // expected-error {{argument value 1024 is outside the valid range [-1024, 1016]}}
  __builtin_lasx_xvstelm_d(_1, _2, var, 1); // expected-error {{argument to '__builtin_lasx_xvstelm_d' must be a constant integer}}
}

void xvstelm_b_idx(v32i8 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_b(_1, _2, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  __builtin_lasx_xvstelm_b(_1, _2, 1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_lasx_xvstelm_b(_1, _2, 1, var); // expected-error {{argument to '__builtin_lasx_xvstelm_b' must be a constant integer}}
}

void xvstelm_h_idx(v16i16 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_h(_1, _2, 2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  __builtin_lasx_xvstelm_h(_1, _2, 2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  __builtin_lasx_xvstelm_h(_1, _2, 2, var); // expected-error {{argument to '__builtin_lasx_xvstelm_h' must be a constant integer}}
}

void xvstelm_w_idx(v8i32 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_w(_1, _2, 4, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  __builtin_lasx_xvstelm_w(_1, _2, 4, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  __builtin_lasx_xvstelm_w(_1, _2, 4, var); // expected-error {{argument to '__builtin_lasx_xvstelm_w' must be a constant integer}}
}

void xvstelm_d_idx(v4i64 _1, void * _2, int var) {
  __builtin_lasx_xvstelm_d(_1, _2, 8, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  __builtin_lasx_xvstelm_d(_1, _2, 8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_lasx_xvstelm_d(_1, _2, 8, var); // expected-error {{argument to '__builtin_lasx_xvstelm_d' must be a constant integer}}
}

v8i32 xvinsve0_w(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvinsve0_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvinsve0_w(_1, _2, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvinsve0_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvinsve0_w' must be a constant integer}}
  return res;
}

v4i64 xvinsve0_d(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvinsve0_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvinsve0_d(_1, _2, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvinsve0_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvinsve0_d' must be a constant integer}}
  return res;
}

v8i32 xvpickve_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvpickve_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve_w(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve_w' must be a constant integer}}
  return res;
}

v4i64 xvpickve_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvpickve_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve_d(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve_d' must be a constant integer}}
  return res;
}

v4i64 xvldi(int var) {
  v4i64 res = __builtin_lasx_xvldi(-4097); // expected-error {{argument value -4097 is outside the valid range [-4096, 4095]}}
  res |= __builtin_lasx_xvldi(4096); // expected-error {{argument value 4096 is outside the valid range [-4096, 4095]}}
  res |= __builtin_lasx_xvldi(var); // expected-error {{argument to '__builtin_lasx_xvldi' must be a constant integer}}
  return res;
}

v8i32 xvinsgr2vr_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvinsgr2vr_w(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvinsgr2vr_w(_1, 1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvinsgr2vr_w(_1, 1, var); // expected-error {{argument to '__builtin_lasx_xvinsgr2vr_w' must be a constant integer}}
  return res;
}

v4i64 xvinsgr2vr_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvinsgr2vr_d(_1, 1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvinsgr2vr_d(_1, 1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvinsgr2vr_d(_1, 1, var); // expected-error {{argument to '__builtin_lasx_xvinsgr2vr_d' must be a constant integer}}
  return res;
}

v32i8 xvldrepl_b(void *_1, int var) {
  v32i8 res = __builtin_lasx_xvldrepl_b(_1, -2049); // expected-error {{argument value -2049 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lasx_xvldrepl_b(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2047]}}
  res |= __builtin_lasx_xvldrepl_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvldrepl_b' must be a constant integer}}
  return res;
}

v16i16 xvldrepl_h(void *_1, int var) {
  v16i16 res = __builtin_lasx_xvldrepl_h(_1, -2050); // expected-error {{argument value -2050 is outside the valid range [-2048, 2046]}}
  res |= __builtin_lasx_xvldrepl_h(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2046]}}
  res |= __builtin_lasx_xvldrepl_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvldrepl_h' must be a constant integer}}
  return res;
}

v8i32 xvldrepl_w(void *_1, int var) {
  v8i32 res = __builtin_lasx_xvldrepl_w(_1, -2052); // expected-error {{argument value -2052 is outside the valid range [-2048, 2044]}}
  res |= __builtin_lasx_xvldrepl_w(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2044]}}
  res |= __builtin_lasx_xvldrepl_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvldrepl_w' must be a constant integer}}
  return res;
}

v4i64 xvldrepl_d(void *_1, int var) {
  v4i64 res = __builtin_lasx_xvldrepl_d(_1, -2056); // expected-error {{argument value -2056 is outside the valid range [-2048, 2040]}}
  res |= __builtin_lasx_xvldrepl_d(_1, 2048); // expected-error {{argument value 2048 is outside the valid range [-2048, 2040]}}
  res |= __builtin_lasx_xvldrepl_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvldrepl_d' must be a constant integer}}
  return res;
}

int xvpickve2gr_w(v8i32 _1, int var) {
  int res = __builtin_lasx_xvpickve2gr_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve2gr_w(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve2gr_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve2gr_w' must be a constant integer}}
  return res;
}

unsigned int xvpickve2gr_wu(v8i32 _1, int var) {
  unsigned int res = __builtin_lasx_xvpickve2gr_wu(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve2gr_wu(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvpickve2gr_wu(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve2gr_wu' must be a constant integer}}
  return res;
}

long xvpickve2gr_d(v4i64 _1, int var) {
  long res = __builtin_lasx_xvpickve2gr_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve2gr_d(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve2gr_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve2gr_d' must be a constant integer}}
  return res;
}

unsigned long int xvpickve2gr_du(v4i64 _1, int var) {
  unsigned long int res = __builtin_lasx_xvpickve2gr_du(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve2gr_du(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res |= __builtin_lasx_xvpickve2gr_du(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve2gr_du' must be a constant integer}}
  return res;
}

v32i8 xvrotri_b(v32i8 _1, int var) {
  v32i8 res = __builtin_lasx_xvrotri_b(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvrotri_b(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res |= __builtin_lasx_xvrotri_b(_1, var); // expected-error {{argument to '__builtin_lasx_xvrotri_b' must be a constant integer}}
  return res;
}

v16i16 xvrotri_h(v16i16 _1, int var) {
  v16i16 res = __builtin_lasx_xvrotri_h(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvrotri_h(_1, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvrotri_h(_1, var); // expected-error {{argument to '__builtin_lasx_xvrotri_h' must be a constant integer}}
  return res;
}

v8i32 xvrotri_w(v8i32 _1, int var) {
  v8i32 res = __builtin_lasx_xvrotri_w(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvrotri_w(_1, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvrotri_w(_1, var); // expected-error {{argument to '__builtin_lasx_xvrotri_w' must be a constant integer}}
  return res;
}

v4i64 xvrotri_d(v4i64 _1, int var) {
  v4i64 res = __builtin_lasx_xvrotri_d(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvrotri_d(_1, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvrotri_d(_1, var); // expected-error {{argument to '__builtin_lasx_xvrotri_d' must be a constant integer}}
  return res;
}

v32i8 xvsrlni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvsrlni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvsrlni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvsrlni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvsrlni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvsrlni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvsrlni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvsrlni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrlni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrlni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlni_d_q' must be a constant integer}}
  return res;
}

v32i8 xvsrlrni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvsrlrni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlrni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrlrni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlrni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvsrlrni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvsrlrni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlrni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrlrni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlrni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvsrlrni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvsrlrni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlrni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrlrni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlrni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvsrlrni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvsrlrni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrlrni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrlrni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrlrni_d_q' must be a constant integer}}
  return res;
}

v32i8 xvssrlni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvssrlni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvssrlni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvssrlni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvssrlni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvssrlni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvssrlni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvssrlni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_d_q' must be a constant integer}}
  return res;
}

v32u8 xvssrlni_bu_h(v32u8 _1, v32i8 _2, int var) {
  v32u8 res = __builtin_lasx_xvssrlni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_bu_h' must be a constant integer}}
  return res;
}

v16u16 xvssrlni_hu_w(v16u16 _1, v16i16 _2, int var) {
  v16u16 res = __builtin_lasx_xvssrlni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_hu_w' must be a constant integer}}
  return res;
}

v8u32 xvssrlni_wu_d(v8u32 _1, v8i32 _2, int var) {
  v8u32 res = __builtin_lasx_xvssrlni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_wu_d' must be a constant integer}}
  return res;
}

v4u64 xvssrlni_du_q(v4u64 _1, v4i64 _2, int var) {
  v4u64 res = __builtin_lasx_xvssrlni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlni_du_q' must be a constant integer}}
  return res;
}

v32i8 xvssrlrni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvssrlrni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlrni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlrni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvssrlrni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvssrlrni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlrni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlrni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvssrlrni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvssrlrni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlrni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlrni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvssrlrni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvssrlrni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlrni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlrni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_d_q' must be a constant integer}}
  return res;
}

v32u8 xvssrlrni_bu_h(v32u8 _1, v32i8 _2, int var) {
  v32u8 res = __builtin_lasx_xvssrlrni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlrni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrlrni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_bu_h' must be a constant integer}}
  return res;
}

v16u16 xvssrlrni_hu_w(v16u16 _1, v16i16 _2, int var) {
  v16u16 res = __builtin_lasx_xvssrlrni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlrni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrlrni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_hu_w' must be a constant integer}}
  return res;
}

v8u32 xvssrlrni_wu_d(v8u32 _1, v8i32 _2, int var) {
  v8u32 res = __builtin_lasx_xvssrlrni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlrni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrlrni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_wu_d' must be a constant integer}}
  return res;
}

v4u64 xvssrlrni_du_q(v4u64 _1, v4i64 _2, int var) {
  v4u64 res = __builtin_lasx_xvssrlrni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlrni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrlrni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrlrni_du_q' must be a constant integer}}
  return res;
}

v32i8 xvsrani_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvsrani_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrani_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrani_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrani_b_h' must be a constant integer}}
  return res;
}

v16i16 xvsrani_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvsrani_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrani_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrani_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrani_h_w' must be a constant integer}}
  return res;
}

v8i32 xvsrani_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvsrani_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrani_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrani_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrani_w_d' must be a constant integer}}
  return res;
}

v4i64 xvsrani_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvsrani_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrani_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrani_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrani_d_q' must be a constant integer}}
  return res;
}

v32i8 xvsrarni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvsrarni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrarni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvsrarni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrarni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvsrarni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvsrarni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrarni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvsrarni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrarni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvsrarni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvsrarni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrarni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvsrarni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrarni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvsrarni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvsrarni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrarni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvsrarni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvsrarni_d_q' must be a constant integer}}
  return res;
}

v32i8 xvssrani_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvssrani_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrani_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrani_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_b_h' must be a constant integer}}
  return res;
}

v16i16 xvssrani_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvssrani_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrani_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrani_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_h_w' must be a constant integer}}
  return res;
}

v8i32 xvssrani_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvssrani_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrani_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrani_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_w_d' must be a constant integer}}
  return res;
}

v4i64 xvssrani_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvssrani_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrani_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrani_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_d_q' must be a constant integer}}
  return res;
}

v32u8 xvssrani_bu_h(v32u8 _1, v32i8 _2, int var) {
  v32u8 res = __builtin_lasx_xvssrani_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrani_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrani_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_bu_h' must be a constant integer}}
  return res;
}

v16u16 xvssrani_hu_w(v16u16 _1, v16i16 _2, int var) {
  v16u16 res = __builtin_lasx_xvssrani_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrani_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrani_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_hu_w' must be a constant integer}}
  return res;
}

v8u32 xvssrani_wu_d(v8u32 _1, v8i32 _2, int var) {
  v8u32 res = __builtin_lasx_xvssrani_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrani_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrani_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_wu_d' must be a constant integer}}
  return res;
}

v4u64 xvssrani_du_q(v4u64 _1, v4i64 _2, int var) {
  v4u64 res = __builtin_lasx_xvssrani_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrani_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrani_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrani_du_q' must be a constant integer}}
  return res;
}

v32i8 xvssrarni_b_h(v32i8 _1, v32i8 _2, int var) {
  v32i8 res = __builtin_lasx_xvssrarni_b_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrarni_b_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrarni_b_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_b_h' must be a constant integer}}
  return res;
}

v16i16 xvssrarni_h_w(v16i16 _1, v16i16 _2, int var) {
  v16i16 res = __builtin_lasx_xvssrarni_h_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrarni_h_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrarni_h_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_h_w' must be a constant integer}}
  return res;
}

v8i32 xvssrarni_w_d(v8i32 _1, v8i32 _2, int var) {
  v8i32 res = __builtin_lasx_xvssrarni_w_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrarni_w_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrarni_w_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_w_d' must be a constant integer}}
  return res;
}

v4i64 xvssrarni_d_q(v4i64 _1, v4i64 _2, int var) {
  v4i64 res = __builtin_lasx_xvssrarni_d_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrarni_d_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrarni_d_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_d_q' must be a constant integer}}
  return res;
}

v32u8 xvssrarni_bu_h(v32u8 _1, v32i8 _2, int var) {
  v32u8 res = __builtin_lasx_xvssrarni_bu_h(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrarni_bu_h(_1, _2, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  res |= __builtin_lasx_xvssrarni_bu_h(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_bu_h' must be a constant integer}}
  return res;
}

v16u16 xvssrarni_hu_w(v16u16 _1, v16i16 _2, int var) {
  v16u16 res = __builtin_lasx_xvssrarni_hu_w(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrarni_hu_w(_1, _2, 32); // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  res |= __builtin_lasx_xvssrarni_hu_w(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_hu_w' must be a constant integer}}
  return res;
}

v8u32 xvssrarni_wu_d(v8u32 _1, v8i32 _2, int var) {
  v8u32 res = __builtin_lasx_xvssrarni_wu_d(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrarni_wu_d(_1, _2, 64); // expected-error {{argument value 64 is outside the valid range [0, 63]}}
  res |= __builtin_lasx_xvssrarni_wu_d(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_wu_d' must be a constant integer}}
  return res;
}

v4u64 xvssrarni_du_q(v4u64 _1, v4i64 _2, int var) {
  v4u64 res = __builtin_lasx_xvssrarni_du_q(_1, _2, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrarni_du_q(_1, _2, 128); // expected-error {{argument value 128 is outside the valid range [0, 127]}}
  res |= __builtin_lasx_xvssrarni_du_q(_1, _2, var); // expected-error {{argument to '__builtin_lasx_xvssrarni_du_q' must be a constant integer}}
  return res;
}

v4f64 xvpickve_d_f(v4f64 _1, int var) {
  v4f64 res = __builtin_lasx_xvpickve_d_f(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 3]}}
  res += __builtin_lasx_xvpickve_d_f(_1, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  res += __builtin_lasx_xvpickve_d_f(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve_d_f' must be a constant integer}}
  return res;
}

v8f32 xvpickve_w_f(v8f32 _1, int var) {
  v8f32 res = __builtin_lasx_xvpickve_w_f(_1, -1); // expected-error {{argument value 4294967295 is outside the valid range [0, 7]}}
  res += __builtin_lasx_xvpickve_w_f(_1, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  res += __builtin_lasx_xvpickve_w_f(_1, var); // expected-error {{argument to '__builtin_lasx_xvpickve_w_f' must be a constant integer}}
  return res;
}

v32i8 xvrepli_b(int var) {
  v32i8 res = __builtin_lasx_xvrepli_b(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_b(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_b(var); // expected-error {{argument to '__builtin_lasx_xvrepli_b' must be a constant integer}}
  return res;
}

v4i64 xvrepli_d(int var) {
  v4i64 res = __builtin_lasx_xvrepli_d(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_d(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_d(var); // expected-error {{argument to '__builtin_lasx_xvrepli_d' must be a constant integer}}
  return res;
}

v16i16 xvrepli_h(int var) {
  v16i16 res = __builtin_lasx_xvrepli_h(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_h(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_h(var); // expected-error {{argument to '__builtin_lasx_xvrepli_h' must be a constant integer}}
  return res;
}

v8i32 xvrepli_w(int var) {
  v8i32 res = __builtin_lasx_xvrepli_w(-513); // expected-error {{argument value -513 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_w(512); // expected-error {{argument value 512 is outside the valid range [-512, 511]}}
  res |= __builtin_lasx_xvrepli_w(var); // expected-error {{argument to '__builtin_lasx_xvrepli_w' must be a constant integer}}
  return res;
}
