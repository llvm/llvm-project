// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lasx -flax-vector-conversions=none -fno-signed-char

// This test file verifies that the macro-defined LASX intrinsic interfaces
// do not require implicit vector type conversions, allowing them to be
// compiled successfully with -flax-vector-conversions=none.
//
// It ensures that the built-in signatures are strictly aligned with the
// types expected by the intrinsic headers.

#include <lasxintrin.h>

__m256i xvslli_b(v32i8 _1) { return __lasx_xvslli_b(_1, 1); }

__m256i xvslli_h(v16i16 _1) { return __lasx_xvslli_h(_1, 1); }

__m256i xvslli_w(v8i32 _1) { return __lasx_xvslli_w(_1, 1); }

__m256i xvslli_d(v4i64 _1) { return __lasx_xvslli_d(_1, 1); }

__m256i xvsrai_b(v32i8 _1) { return __lasx_xvsrai_b(_1, 1); }

__m256i xvsrai_h(v16i16 _1) { return __lasx_xvsrai_h(_1, 1); }

__m256i xvsrai_w(v8i32 _1) { return __lasx_xvsrai_w(_1, 1); }

__m256i xvsrai_d(v4i64 _1) { return __lasx_xvsrai_d(_1, 1); }

__m256i xvsrari_b(v32i8 _1) { return __lasx_xvsrari_b(_1, 1); }

__m256i xvsrari_h(v16i16 _1) { return __lasx_xvsrari_h(_1, 1); }

__m256i xvsrari_w(v8i32 _1) { return __lasx_xvsrari_w(_1, 1); }

__m256i xvsrari_d(v4i64 _1) { return __lasx_xvsrari_d(_1, 1); }

__m256i xvsrli_b(v32i8 _1) { return __lasx_xvsrli_b(_1, 1); }

__m256i xvsrli_h(v16i16 _1) { return __lasx_xvsrli_h(_1, 1); }

__m256i xvsrli_w(v8i32 _1) { return __lasx_xvsrli_w(_1, 1); }

__m256i xvsrli_d(v4i64 _1) { return __lasx_xvsrli_d(_1, 1); }

__m256i xvsrlri_b(v32i8 _1) { return __lasx_xvsrlri_b(_1, 1); }

__m256i xvsrlri_h(v16i16 _1) { return __lasx_xvsrlri_h(_1, 1); }

__m256i xvsrlri_w(v8i32 _1) { return __lasx_xvsrlri_w(_1, 1); }

__m256i xvsrlri_d(v4i64 _1) { return __lasx_xvsrlri_d(_1, 1); }

__m256i xvbitclri_b(v32u8 _1) { return __lasx_xvbitclri_b(_1, 1); }

__m256i xvbitclri_h(v16u16 _1) { return __lasx_xvbitclri_h(_1, 1); }

__m256i xvbitclri_w(v8u32 _1) { return __lasx_xvbitclri_w(_1, 1); }

__m256i xvbitclri_d(v4u64 _1) { return __lasx_xvbitclri_d(_1, 1); }

__m256i xvbitseti_b(v32u8 _1) { return __lasx_xvbitseti_b(_1, 1); }

__m256i xvbitseti_h(v16u16 _1) { return __lasx_xvbitseti_h(_1, 1); }

__m256i xvbitseti_w(v8u32 _1) { return __lasx_xvbitseti_w(_1, 1); }

__m256i xvbitseti_d(v4u64 _1) { return __lasx_xvbitseti_d(_1, 1); }

__m256i xvbitrevi_b(v32u8 _1) { return __lasx_xvbitrevi_b(_1, 1); }

__m256i xvbitrevi_h(v16u16 _1) { return __lasx_xvbitrevi_h(_1, 1); }

__m256i xvbitrevi_w(v8u32 _1) { return __lasx_xvbitrevi_w(_1, 1); }

__m256i xvbitrevi_d(v4u64 _1) { return __lasx_xvbitrevi_d(_1, 1); }

__m256i xvaddi_bu(v32i8 _1) { return __lasx_xvaddi_bu(_1, 1); }

__m256i xvaddi_hu(v16i16 _1) { return __lasx_xvaddi_hu(_1, 1); }

__m256i xvaddi_wu(v8i32 _1) { return __lasx_xvaddi_wu(_1, 1); }

__m256i xvaddi_du(v4i64 _1) { return __lasx_xvaddi_du(_1, 1); }

__m256i xvsubi_bu(v32i8 _1) { return __lasx_xvsubi_bu(_1, 1); }

__m256i xvsubi_hu(v16i16 _1) { return __lasx_xvsubi_hu(_1, 1); }

__m256i xvsubi_wu(v8i32 _1) { return __lasx_xvsubi_wu(_1, 1); }

__m256i xvsubi_du(v4i64 _1) { return __lasx_xvsubi_du(_1, 1); }

__m256i xvmaxi_b(v32i8 _1) { return __lasx_xvmaxi_b(_1, 1); }

__m256i xvmaxi_h(v16i16 _1) { return __lasx_xvmaxi_h(_1, 1); }

__m256i xvmaxi_w(v8i32 _1) { return __lasx_xvmaxi_w(_1, 1); }

__m256i xvmaxi_d(v4i64 _1) { return __lasx_xvmaxi_d(_1, 1); }

__m256i xvmaxi_bu(v32u8 _1) { return __lasx_xvmaxi_bu(_1, 1); }

__m256i xvmaxi_hu(v16u16 _1) { return __lasx_xvmaxi_hu(_1, 1); }

__m256i xvmaxi_wu(v8u32 _1) { return __lasx_xvmaxi_wu(_1, 1); }

__m256i xvmaxi_du(v4u64 _1) { return __lasx_xvmaxi_du(_1, 1); }

__m256i xvmini_b(v32i8 _1) { return __lasx_xvmini_b(_1, 1); }

__m256i xvmini_h(v16i16 _1) { return __lasx_xvmini_h(_1, 1); }

__m256i xvmini_w(v8i32 _1) { return __lasx_xvmini_w(_1, 1); }

__m256i xvmini_d(v4i64 _1) { return __lasx_xvmini_d(_1, 1); }

__m256i xvmini_bu(v32u8 _1) { return __lasx_xvmini_bu(_1, 1); }

__m256i xvmini_hu(v16u16 _1) { return __lasx_xvmini_hu(_1, 1); }

__m256i xvmini_wu(v8u32 _1) { return __lasx_xvmini_wu(_1, 1); }

__m256i xvmini_du(v4u64 _1) { return __lasx_xvmini_du(_1, 1); }

__m256i xvseqi_b(v32i8 _1) { return __lasx_xvseqi_b(_1, 1); }

__m256i xvseqi_h(v16i16 _1) { return __lasx_xvseqi_h(_1, 1); }

__m256i xvseqi_w(v8i32 _1) { return __lasx_xvseqi_w(_1, 1); }

__m256i xvseqi_d(v4i64 _1) { return __lasx_xvseqi_d(_1, 1); }

__m256i xvslti_b(v32i8 _1) { return __lasx_xvslti_b(_1, 1); }

__m256i xvslti_h(v16i16 _1) { return __lasx_xvslti_h(_1, 1); }

__m256i xvslti_w(v8i32 _1) { return __lasx_xvslti_w(_1, 1); }

__m256i xvslti_d(v4i64 _1) { return __lasx_xvslti_d(_1, 1); }

__m256i xvslti_bu(v32u8 _1) { return __lasx_xvslti_bu(_1, 1); }

__m256i xvslti_hu(v16u16 _1) { return __lasx_xvslti_hu(_1, 1); }

__m256i xvslti_wu(v8u32 _1) { return __lasx_xvslti_wu(_1, 1); }

__m256i xvslti_du(v4u64 _1) { return __lasx_xvslti_du(_1, 1); }

__m256i xvslei_b(v32i8 _1) { return __lasx_xvslei_b(_1, 1); }

__m256i xvslei_h(v16i16 _1) { return __lasx_xvslei_h(_1, 1); }

__m256i xvslei_w(v8i32 _1) { return __lasx_xvslei_w(_1, 1); }

__m256i xvslei_d(v4i64 _1) { return __lasx_xvslei_d(_1, 1); }

__m256i xvslei_bu(v32u8 _1) { return __lasx_xvslei_bu(_1, 1); }

__m256i xvslei_hu(v16u16 _1) { return __lasx_xvslei_hu(_1, 1); }

__m256i xvslei_wu(v8u32 _1) { return __lasx_xvslei_wu(_1, 1); }

__m256i xvslei_du(v4u64 _1) { return __lasx_xvslei_du(_1, 1); }

__m256i xvsat_b(v32i8 _1) { return __lasx_xvsat_b(_1, 1); }

__m256i xvsat_h(v16i16 _1) { return __lasx_xvsat_h(_1, 1); }

__m256i xvsat_w(v8i32 _1) { return __lasx_xvsat_w(_1, 1); }

__m256i xvsat_d(v4i64 _1) { return __lasx_xvsat_d(_1, 1); }

__m256i xvsat_bu(v32u8 _1) { return __lasx_xvsat_bu(_1, 1); }

__m256i xvsat_hu(v16u16 _1) { return __lasx_xvsat_hu(_1, 1); }

__m256i xvsat_wu(v8u32 _1) { return __lasx_xvsat_wu(_1, 1); }

__m256i xvsat_du(v4u64 _1) { return __lasx_xvsat_du(_1, 1); }

__m256i xvrepl128vei_b(v32i8 _1) { return __lasx_xvrepl128vei_b(_1, 1); }

__m256i xvrepl128vei_h(v16i16 _1) { return __lasx_xvrepl128vei_h(_1, 1); }

__m256i xvrepl128vei_w(v8i32 _1) { return __lasx_xvrepl128vei_w(_1, 1); }

__m256i xvrepl128vei_d(v4i64 _1) { return __lasx_xvrepl128vei_d(_1, 1); }

__m256i xvandi_b(v32u8 _1) { return __lasx_xvandi_b(_1, 1); }

__m256i xvori_b(v32u8 _1) { return __lasx_xvori_b(_1, 1); }

__m256i xvnori_b(v32u8 _1) { return __lasx_xvnori_b(_1, 1); }

__m256i xvxori_b(v32u8 _1) { return __lasx_xvxori_b(_1, 1); }

__m256i xvbitseli_b(v32u8 _1, v32u8 _2) {
  return __lasx_xvbitseli_b(_1, _2, 1);
}

__m256i xvshuf4i_b(v32i8 _1) { return __lasx_xvshuf4i_b(_1, 1); }

__m256i xvshuf4i_h(v16i16 _1) { return __lasx_xvshuf4i_h(_1, 1); }

__m256i xvshuf4i_w(v8i32 _1) { return __lasx_xvshuf4i_w(_1, 1); }

__m256i xvpermi_w(v8i32 _1, v8i32 _2) { return __lasx_xvpermi_w(_1, _2, 1); }

__m256i xvsllwil_h_b(v32i8 _1) { return __lasx_xvsllwil_h_b(_1, 1); }

__m256i xvsllwil_w_h(v16i16 _1) { return __lasx_xvsllwil_w_h(_1, 1); }

__m256i xvsllwil_d_w(v8i32 _1) { return __lasx_xvsllwil_d_w(_1, 1); }

__m256i xvsllwil_hu_bu(v32u8 _1) { return __lasx_xvsllwil_hu_bu(_1, 1); }

__m256i xvsllwil_wu_hu(v16u16 _1) { return __lasx_xvsllwil_wu_hu(_1, 1); }

__m256i xvsllwil_du_wu(v8u32 _1) { return __lasx_xvsllwil_du_wu(_1, 1); }

__m256i xvfrstpi_b(v32i8 _1, v32i8 _2) { return __lasx_xvfrstpi_b(_1, _2, 1); }

__m256i xvfrstpi_h(v16i16 _1, v16i16 _2) {
  return __lasx_xvfrstpi_h(_1, _2, 1);
}

__m256i xvshuf4i_d(v4i64 _1, v4i64 _2) { return __lasx_xvshuf4i_d(_1, _2, 1); }

__m256i xvbsrl_v(v32i8 _1) { return __lasx_xvbsrl_v(_1, 1); }

__m256i xvbsll_v(v32i8 _1) { return __lasx_xvbsll_v(_1, 1); }

__m256i xvextrins_b(v32i8 _1, v32i8 _2) {
  return __lasx_xvextrins_b(_1, _2, 1);
}

__m256i xvextrins_h(v16i16 _1, v16i16 _2) {
  return __lasx_xvextrins_h(_1, _2, 1);
}

__m256i xvextrins_w(v8i32 _1, v8i32 _2) {
  return __lasx_xvextrins_w(_1, _2, 1);
}

__m256i xvextrins_d(v4i64 _1, v4i64 _2) {
  return __lasx_xvextrins_d(_1, _2, 1);
}

__m256i xvld(void *_1) { return __lasx_xvld(_1, 1); }

void xvst(v32i8 _1, void *_2) { return __lasx_xvst(_1, _2, 1); }

void xvstelm_b(v32i8 _1, void *_2) { return __lasx_xvstelm_b(_1, _2, 1, 1); }

void xvstelm_h(v16i16 _1, void *_2) { return __lasx_xvstelm_h(_1, _2, 1, 1); }

void xvstelm_w(v8i32 _1, void *_2) { return __lasx_xvstelm_w(_1, _2, 1, 1); }

void xvstelm_d(v4i64 _1, void *_2) { return __lasx_xvstelm_d(_1, _2, 1, 1); }

__m256i xvinsve0_w(v8i32 _1, v8i32 _2) { return __lasx_xvinsve0_w(_1, _2, 1); }

__m256i xvinsve0_d(v4i64 _1, v4i64 _2) { return __lasx_xvinsve0_d(_1, _2, 1); }

__m256i xvpickve_w(v8i32 _1) { return __lasx_xvpickve_w(_1, 1); }

__m256i xvpickve_d(v4i64 _1) { return __lasx_xvpickve_d(_1, 1); }

__m256i xvldi() { return __lasx_xvldi(1); }

__m256i xvinsgr2vr_w(v8i32 _1) { return __lasx_xvinsgr2vr_w(_1, 1, 1); }

__m256i xvinsgr2vr_d(v4i64 _1) { return __lasx_xvinsgr2vr_d(_1, 1, 1); }

__m256i xvpermi_q(v32i8 _1, v32i8 _2) { return __lasx_xvpermi_q(_1, _2, 1); }

__m256i xvpermi_d(v4i64 _1) { return __lasx_xvpermi_d(_1, 1); }

__m256i xvldrepl_b(void *_1) { return __lasx_xvldrepl_b(_1, 1); }

__m256i xvldrepl_h(void *_1) { return __lasx_xvldrepl_h(_1, 1); }

__m256i xvldrepl_w(void *_1) { return __lasx_xvldrepl_w(_1, 1); }

__m256i xvldrepl_d(void *_1) { return __lasx_xvldrepl_d(_1, 1); }

int xvpickve2gr_w(v8i32 _1) { return __lasx_xvpickve2gr_w(_1, 1); }

unsigned int xvpickve2gr_wu(v8i32 _1) { return __lasx_xvpickve2gr_wu(_1, 1); }

long int xvpickve2gr_d(v4i64 _1) { return __lasx_xvpickve2gr_d(_1, 1); }

unsigned long int xvpickve2gr_du(v4i64 _1) {
  return __lasx_xvpickve2gr_du(_1, 1);
}

__m256i xvrotri_b(v32i8 _1) { return __lasx_xvrotri_b(_1, 1); }

__m256i xvrotri_h(v16i16 _1) { return __lasx_xvrotri_h(_1, 1); }

__m256i xvrotri_w(v8i32 _1) { return __lasx_xvrotri_w(_1, 1); }

__m256i xvrotri_d(v4i64 _1) { return __lasx_xvrotri_d(_1, 1); }

__m256i xvsrlni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvsrlni_b_h(_1, _2, 1);
}

__m256i xvsrlni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvsrlni_h_w(_1, _2, 1);
}

__m256i xvsrlni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvsrlni_w_d(_1, _2, 1);
}

__m256i xvsrlni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvsrlni_d_q(_1, _2, 1);
}

__m256i xvsrlrni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvsrlrni_b_h(_1, _2, 1);
}

__m256i xvsrlrni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvsrlrni_h_w(_1, _2, 1);
}

__m256i xvsrlrni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvsrlrni_w_d(_1, _2, 1);
}

__m256i xvsrlrni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvsrlrni_d_q(_1, _2, 1);
}

__m256i xvssrlni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvssrlni_b_h(_1, _2, 1);
}

__m256i xvssrlni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvssrlni_h_w(_1, _2, 1);
}

__m256i xvssrlni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvssrlni_w_d(_1, _2, 1);
}

__m256i xvssrlni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvssrlni_d_q(_1, _2, 1);
}

__m256i xvssrlni_bu_h(v32u8 _1, v32i8 _2) {
  return __lasx_xvssrlni_bu_h(_1, _2, 1);
}

__m256i xvssrlni_hu_w(v16u16 _1, v16i16 _2) {
  return __lasx_xvssrlni_hu_w(_1, _2, 1);
}

__m256i xvssrlni_wu_d(v8u32 _1, v8i32 _2) {
  return __lasx_xvssrlni_wu_d(_1, _2, 1);
}

__m256i xvssrlni_du_q(v4u64 _1, v4i64 _2) {
  return __lasx_xvssrlni_du_q(_1, _2, 1);
}

__m256i xvssrlrni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvssrlrni_b_h(_1, _2, 1);
}

__m256i xvssrlrni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvssrlrni_h_w(_1, _2, 1);
}

__m256i xvssrlrni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvssrlrni_w_d(_1, _2, 1);
}

__m256i xvssrlrni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvssrlrni_d_q(_1, _2, 1);
}

__m256i xvssrlrni_bu_h(v32u8 _1, v32i8 _2) {
  return __lasx_xvssrlrni_bu_h(_1, _2, 1);
}

__m256i xvssrlrni_hu_w(v16u16 _1, v16i16 _2) {
  return __lasx_xvssrlrni_hu_w(_1, _2, 1);
}

__m256i xvssrlrni_wu_d(v8u32 _1, v8i32 _2) {
  return __lasx_xvssrlrni_wu_d(_1, _2, 1);
}

__m256i xvssrlrni_du_q(v4u64 _1, v4i64 _2) {
  return __lasx_xvssrlrni_du_q(_1, _2, 1);
}

__m256i xvsrani_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvsrani_b_h(_1, _2, 1);
}

__m256i xvsrani_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvsrani_h_w(_1, _2, 1);
}

__m256i xvsrani_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvsrani_w_d(_1, _2, 1);
}

__m256i xvsrani_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvsrani_d_q(_1, _2, 1);
}

__m256i xvsrarni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvsrarni_b_h(_1, _2, 1);
}

__m256i xvsrarni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvsrarni_h_w(_1, _2, 1);
}

__m256i xvsrarni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvsrarni_w_d(_1, _2, 1);
}

__m256i xvsrarni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvsrarni_d_q(_1, _2, 1);
}

__m256i xvssrani_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvssrani_b_h(_1, _2, 1);
}

__m256i xvssrani_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvssrani_h_w(_1, _2, 1);
}

__m256i xvssrani_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvssrani_w_d(_1, _2, 1);
}

__m256i xvssrani_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvssrani_d_q(_1, _2, 1);
}

__m256i xvssrani_bu_h(v32u8 _1, v32i8 _2) {
  return __lasx_xvssrani_bu_h(_1, _2, 1);
}

__m256i xvssrani_hu_w(v16u16 _1, v16i16 _2) {
  return __lasx_xvssrani_hu_w(_1, _2, 1);
}

__m256i xvssrani_wu_d(v8u32 _1, v8i32 _2) {
  return __lasx_xvssrani_wu_d(_1, _2, 1);
}

__m256i xvssrani_du_q(v4u64 _1, v4i64 _2) {
  return __lasx_xvssrani_du_q(_1, _2, 1);
}

__m256i xvssrarni_b_h(v32i8 _1, v32i8 _2) {
  return __lasx_xvssrarni_b_h(_1, _2, 1);
}

__m256i xvssrarni_h_w(v16i16 _1, v16i16 _2) {
  return __lasx_xvssrarni_h_w(_1, _2, 1);
}

__m256i xvssrarni_w_d(v8i32 _1, v8i32 _2) {
  return __lasx_xvssrarni_w_d(_1, _2, 1);
}

__m256i xvssrarni_d_q(v4i64 _1, v4i64 _2) {
  return __lasx_xvssrarni_d_q(_1, _2, 1);
}

__m256i xvssrarni_bu_h(v32u8 _1, v32i8 _2) {
  return __lasx_xvssrarni_bu_h(_1, _2, 1);
}

__m256i xvssrarni_hu_w(v16u16 _1, v16i16 _2) {
  return __lasx_xvssrarni_hu_w(_1, _2, 1);
}

__m256i xvssrarni_wu_d(v8u32 _1, v8i32 _2) {
  return __lasx_xvssrarni_wu_d(_1, _2, 1);
}

__m256i xvssrarni_du_q(v4u64 _1, v4i64 _2) {
  return __lasx_xvssrarni_du_q(_1, _2, 1);
}

int xbnz_b(v32u8 _1) { return __lasx_xbnz_b(_1); }

int xbnz_d(v4u64 _1) { return __lasx_xbnz_d(_1); }

int xbnz_h(v16u16 _1) { return __lasx_xbnz_h(_1); }

int xbnz_v(v32u8 _1) { return __lasx_xbnz_v(_1); }

int xbnz_w(v8u32 _1) { return __lasx_xbnz_w(_1); }

int xbz_b(v32u8 _1) { return __lasx_xbz_b(_1); }

int xbz_d(v4u64 _1) { return __lasx_xbz_d(_1); }

int xbz_h(v16u16 _1) { return __lasx_xbz_h(_1); }

int xbz_v(v32u8 _1) { return __lasx_xbz_v(_1); }

int xbz_w(v8u32 _1) { return __lasx_xbz_w(_1); }

__m256d xvpickve_d_f(v4f64 _1) { return __lasx_xvpickve_d_f(_1, 1); }

__m256 xvpickve_w_f(v8f32 _1) { return __lasx_xvpickve_w_f(_1, 1); }

__m256i xvrepli_b() { return __lasx_xvrepli_b(1); }

__m256i xvrepli_d() { return __lasx_xvrepli_d(1); }

__m256i xvrepli_h() { return __lasx_xvrepli_h(1); }

__m256i xvrepli_w() { return __lasx_xvrepli_w(1); }
