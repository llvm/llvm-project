
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx -flax-vector-conversions=none
// RUN: %clang_cc1 %s -fsyntax-only -triple loongarch64 -target-feature +lsx -flax-vector-conversions=none -fno-signed-char

// This test file verifies that the macro-defined LSX intrinsic interfaces
// do not require implicit vector type conversions, allowing them to be
// compiled successfully with -flax-vector-conversions=none.
//
// It ensures that the built-in signatures are strictly aligned with the
// types expected by the intrinsic headers.

#include <lsxintrin.h>

__m128i vslli_b(v16i8 _1) { return __lsx_vslli_b(_1, 1); }

__m128i vslli_h(v8i16 _1) { return __lsx_vslli_h(_1, 1); }

__m128i vslli_w(v4i32 _1) { return __lsx_vslli_w(_1, 1); }

__m128i vslli_d(v2i64 _1) { return __lsx_vslli_d(_1, 1); }

__m128i vsrai_b(v16i8 _1) { return __lsx_vsrai_b(_1, 1); }

__m128i vsrai_h(v8i16 _1) { return __lsx_vsrai_h(_1, 1); }

__m128i vsrai_w(v4i32 _1) { return __lsx_vsrai_w(_1, 1); }

__m128i vsrai_d(v2i64 _1) { return __lsx_vsrai_d(_1, 1); }

__m128i vsrari_b(v16i8 _1) { return __lsx_vsrari_b(_1, 1); }

__m128i vsrari_h(v8i16 _1) { return __lsx_vsrari_h(_1, 1); }

__m128i vsrari_w(v4i32 _1) { return __lsx_vsrari_w(_1, 1); }

__m128i vsrari_d(v2i64 _1) { return __lsx_vsrari_d(_1, 1); }

__m128i vsrli_b(v16i8 _1) { return __lsx_vsrli_b(_1, 1); }

__m128i vsrli_h(v8i16 _1) { return __lsx_vsrli_h(_1, 1); }

__m128i vsrli_w(v4i32 _1) { return __lsx_vsrli_w(_1, 1); }

__m128i vsrli_d(v2i64 _1) { return __lsx_vsrli_d(_1, 1); }

__m128i vsrlri_b(v16i8 _1) { return __lsx_vsrlri_b(_1, 1); }

__m128i vsrlri_h(v8i16 _1) { return __lsx_vsrlri_h(_1, 1); }

__m128i vsrlri_w(v4i32 _1) { return __lsx_vsrlri_w(_1, 1); }

__m128i vsrlri_d(v2i64 _1) { return __lsx_vsrlri_d(_1, 1); }

__m128i vbitclri_b(v16u8 _1) { return __lsx_vbitclri_b(_1, 1); }

__m128i vbitclri_h(v8u16 _1) { return __lsx_vbitclri_h(_1, 1); }

__m128i vbitclri_w(v4u32 _1) { return __lsx_vbitclri_w(_1, 1); }

__m128i vbitclri_d(v2u64 _1) { return __lsx_vbitclri_d(_1, 1); }

__m128i vbitseti_b(v16u8 _1) { return __lsx_vbitseti_b(_1, 1); }

__m128i vbitseti_h(v8u16 _1) { return __lsx_vbitseti_h(_1, 1); }

__m128i vbitseti_w(v4u32 _1) { return __lsx_vbitseti_w(_1, 1); }

__m128i vbitseti_d(v2u64 _1) { return __lsx_vbitseti_d(_1, 1); }

__m128i vbitrevi_b(v16u8 _1) { return __lsx_vbitrevi_b(_1, 1); }

__m128i vbitrevi_h(v8u16 _1) { return __lsx_vbitrevi_h(_1, 1); }

__m128i vbitrevi_w(v4u32 _1) { return __lsx_vbitrevi_w(_1, 1); }

__m128i vbitrevi_d(v2u64 _1) { return __lsx_vbitrevi_d(_1, 1); }

__m128i vaddi_bu(v16i8 _1) { return __lsx_vaddi_bu(_1, 1); }

__m128i vaddi_hu(v8i16 _1) { return __lsx_vaddi_hu(_1, 1); }

__m128i vaddi_wu(v4i32 _1) { return __lsx_vaddi_wu(_1, 1); }

__m128i vaddi_du(v2i64 _1) { return __lsx_vaddi_du(_1, 1); }

__m128i vsubi_bu(v16i8 _1) { return __lsx_vsubi_bu(_1, 1); }

__m128i vsubi_hu(v8i16 _1) { return __lsx_vsubi_hu(_1, 1); }

__m128i vsubi_wu(v4i32 _1) { return __lsx_vsubi_wu(_1, 1); }

__m128i vsubi_du(v2i64 _1) { return __lsx_vsubi_du(_1, 1); }

__m128i vmaxi_b(v16i8 _1) { return __lsx_vmaxi_b(_1, 1); }

__m128i vmaxi_h(v8i16 _1) { return __lsx_vmaxi_h(_1, 1); }

__m128i vmaxi_w(v4i32 _1) { return __lsx_vmaxi_w(_1, 1); }

__m128i vmaxi_d(v2i64 _1) { return __lsx_vmaxi_d(_1, 1); }

__m128i vmaxi_bu(v16u8 _1) { return __lsx_vmaxi_bu(_1, 1); }

__m128i vmaxi_hu(v8u16 _1) { return __lsx_vmaxi_hu(_1, 1); }

__m128i vmaxi_wu(v4u32 _1) { return __lsx_vmaxi_wu(_1, 1); }

__m128i vmaxi_du(v2u64 _1) { return __lsx_vmaxi_du(_1, 1); }

__m128i vmini_b(v16i8 _1) { return __lsx_vmini_b(_1, 1); }

__m128i vmini_h(v8i16 _1) { return __lsx_vmini_h(_1, 1); }

__m128i vmini_w(v4i32 _1) { return __lsx_vmini_w(_1, 1); }

__m128i vmini_d(v2i64 _1) { return __lsx_vmini_d(_1, 1); }

__m128i vmini_bu(v16u8 _1) { return __lsx_vmini_bu(_1, 1); }

__m128i vmini_hu(v8u16 _1) { return __lsx_vmini_hu(_1, 1); }

__m128i vmini_wu(v4u32 _1) { return __lsx_vmini_wu(_1, 1); }

__m128i vmini_du(v2u64 _1) { return __lsx_vmini_du(_1, 1); }

__m128i vseqi_b(v16i8 _1) { return __lsx_vseqi_b(_1, 1); }

__m128i vseqi_h(v8i16 _1) { return __lsx_vseqi_h(_1, 1); }

__m128i vseqi_w(v4i32 _1) { return __lsx_vseqi_w(_1, 1); }

__m128i vseqi_d(v2i64 _1) { return __lsx_vseqi_d(_1, 1); }

__m128i vslti_b(v16i8 _1) { return __lsx_vslti_b(_1, 1); }

__m128i vslti_h(v8i16 _1) { return __lsx_vslti_h(_1, 1); }

__m128i vslti_w(v4i32 _1) { return __lsx_vslti_w(_1, 1); }

__m128i vslti_d(v2i64 _1) { return __lsx_vslti_d(_1, 1); }

__m128i vslti_bu(v16u8 _1) { return __lsx_vslti_bu(_1, 1); }

__m128i vslti_hu(v8u16 _1) { return __lsx_vslti_hu(_1, 1); }

__m128i vslti_wu(v4u32 _1) { return __lsx_vslti_wu(_1, 1); }

__m128i vslti_du(v2u64 _1) { return __lsx_vslti_du(_1, 1); }

__m128i vslei_b(v16i8 _1) { return __lsx_vslei_b(_1, 1); }

__m128i vslei_h(v8i16 _1) { return __lsx_vslei_h(_1, 1); }

__m128i vslei_w(v4i32 _1) { return __lsx_vslei_w(_1, 1); }

__m128i vslei_d(v2i64 _1) { return __lsx_vslei_d(_1, 1); }

__m128i vslei_bu(v16u8 _1) { return __lsx_vslei_bu(_1, 1); }

__m128i vslei_hu(v8u16 _1) { return __lsx_vslei_hu(_1, 1); }

__m128i vslei_wu(v4u32 _1) { return __lsx_vslei_wu(_1, 1); }

__m128i vslei_du(v2u64 _1) { return __lsx_vslei_du(_1, 1); }

__m128i vsat_b(v16i8 _1) { return __lsx_vsat_b(_1, 1); }

__m128i vsat_h(v8i16 _1) { return __lsx_vsat_h(_1, 1); }

__m128i vsat_w(v4i32 _1) { return __lsx_vsat_w(_1, 1); }

__m128i vsat_d(v2i64 _1) { return __lsx_vsat_d(_1, 1); }

__m128i vsat_bu(v16u8 _1) { return __lsx_vsat_bu(_1, 1); }

__m128i vsat_hu(v8u16 _1) { return __lsx_vsat_hu(_1, 1); }

__m128i vsat_wu(v4u32 _1) { return __lsx_vsat_wu(_1, 1); }

__m128i vsat_du(v2u64 _1) { return __lsx_vsat_du(_1, 1); }

__m128i vreplvei_b(v16i8 _1) { return __lsx_vreplvei_b(_1, 1); }

__m128i vreplvei_h(v8i16 _1) { return __lsx_vreplvei_h(_1, 1); }

__m128i vreplvei_w(v4i32 _1) { return __lsx_vreplvei_w(_1, 1); }

__m128i vreplvei_d(v2i64 _1) { return __lsx_vreplvei_d(_1, 1); }

__m128i vandi_b(v16u8 _1) { return __lsx_vandi_b(_1, 1); }

__m128i vori_b(v16u8 _1) { return __lsx_vori_b(_1, 1); }

__m128i vnori_b(v16u8 _1) { return __lsx_vnori_b(_1, 1); }

__m128i vxori_b(v16u8 _1) { return __lsx_vxori_b(_1, 1); }

__m128i vbitseli_b(v16u8 _1, v16u8 _2) { return __lsx_vbitseli_b(_1, _2, 1); }

__m128i vshuf4i_b(v16i8 _1) { return __lsx_vshuf4i_b(_1, 1); }

__m128i vshuf4i_h(v8i16 _1) { return __lsx_vshuf4i_h(_1, 1); }

__m128i vshuf4i_w(v4i32 _1) { return __lsx_vshuf4i_w(_1, 1); }

int vpickve2gr_b(v16i8 _1) { return __lsx_vpickve2gr_b(_1, 1); }

int vpickve2gr_h(v8i16 _1) { return __lsx_vpickve2gr_h(_1, 1); }

int vpickve2gr_w(v4i32 _1) { return __lsx_vpickve2gr_w(_1, 1); }

long int vpickve2gr_d(v2i64 _1) { return __lsx_vpickve2gr_d(_1, 1); }

unsigned int vpickve2gr_bu(v16i8 _1) { return __lsx_vpickve2gr_bu(_1, 1); }

unsigned int vpickve2gr_hu(v8i16 _1) { return __lsx_vpickve2gr_hu(_1, 1); }

unsigned int vpickve2gr_wu(v4i32 _1) { return __lsx_vpickve2gr_wu(_1, 1); }

unsigned long int vpickve2gr_du(v2i64 _1) { return __lsx_vpickve2gr_du(_1, 1); }

__m128i vinsgr2vr_b(v16i8 _1) { return __lsx_vinsgr2vr_b(_1, 1, 1); }

__m128i vinsgr2vr_h(v8i16 _1) { return __lsx_vinsgr2vr_h(_1, 1, 1); }

__m128i vinsgr2vr_w(v4i32 _1) { return __lsx_vinsgr2vr_w(_1, 1, 1); }

__m128i vinsgr2vr_d(v2i64 _1) { return __lsx_vinsgr2vr_d(_1, 1, 1); }

__m128i vsllwil_h_b(v16i8 _1) { return __lsx_vsllwil_h_b(_1, 1); }

__m128i vsllwil_w_h(v8i16 _1) { return __lsx_vsllwil_w_h(_1, 1); }

__m128i vsllwil_d_w(v4i32 _1) { return __lsx_vsllwil_d_w(_1, 1); }

__m128i vsllwil_hu_bu(v16u8 _1) { return __lsx_vsllwil_hu_bu(_1, 1); }

__m128i vsllwil_wu_hu(v8u16 _1) { return __lsx_vsllwil_wu_hu(_1, 1); }

__m128i vsllwil_du_wu(v4u32 _1) { return __lsx_vsllwil_du_wu(_1, 1); }

__m128i vfrstpi_b(v16i8 _1, v16i8 _2) { return __lsx_vfrstpi_b(_1, _2, 1); }

__m128i vfrstpi_h(v8i16 _1, v8i16 _2) { return __lsx_vfrstpi_h(_1, _2, 1); }

__m128i vshuf4i_d(v2i64 _1, v2i64 _2) { return __lsx_vshuf4i_d(_1, _2, 1); }

__m128i vbsrl_v(v16i8 _1) { return __lsx_vbsrl_v(_1, 1); }

__m128i vbsll_v(v16i8 _1) { return __lsx_vbsll_v(_1, 1); }

__m128i vextrins_b(v16i8 _1, v16i8 _2) { return __lsx_vextrins_b(_1, _2, 1); }

__m128i vextrins_h(v8i16 _1, v8i16 _2) { return __lsx_vextrins_h(_1, _2, 1); }

__m128i vextrins_w(v4i32 _1, v4i32 _2) { return __lsx_vextrins_w(_1, _2, 1); }

__m128i vextrins_d(v2i64 _1, v2i64 _2) { return __lsx_vextrins_d(_1, _2, 1); }

void vstelm_b(v16i8 _1, void *_2) { return __lsx_vstelm_b(_1, _2, 1, 1); }

void vstelm_h(v8i16 _1, void *_2) { return __lsx_vstelm_h(_1, _2, 2, 1); }

void vstelm_w(v4i32 _1, void *_2) { return __lsx_vstelm_w(_1, _2, 4, 1); }

void vstelm_d(v2i64 _1, void *_2) { return __lsx_vstelm_d(_1, _2, 8, 1); }

__m128i vldrepl_b(void *_1) { return __lsx_vldrepl_b(_1, 1); }

__m128i vldrepl_h(void *_1) { return __lsx_vldrepl_h(_1, 1); }

__m128i vldrepl_w(void *_1) { return __lsx_vldrepl_w(_1, 1); }

__m128i vldrepl_d(void *_1) { return __lsx_vldrepl_d(_1, 1); }

__m128i vrotri_b(v16i8 _1) { return __lsx_vrotri_b(_1, 1); }

__m128i vrotri_h(v8i16 _1) { return __lsx_vrotri_h(_1, 1); }

__m128i vrotri_w(v4i32 _1) { return __lsx_vrotri_w(_1, 1); }

__m128i vrotri_d(v2i64 _1) { return __lsx_vrotri_d(_1, 1); }

__m128i vsrlni_b_h(v16i8 _1, v16i8 _2) { return __lsx_vsrlni_b_h(_1, _2, 1); }

__m128i vsrlni_h_w(v8i16 _1, v8i16 _2) { return __lsx_vsrlni_h_w(_1, _2, 1); }

__m128i vsrlni_w_d(v4i32 _1, v4i32 _2) { return __lsx_vsrlni_w_d(_1, _2, 1); }

__m128i vsrlni_d_q(v2i64 _1, v2i64 _2) { return __lsx_vsrlni_d_q(_1, _2, 1); }

__m128i vsrlrni_b_h(v16i8 _1, v16i8 _2) { return __lsx_vsrlrni_b_h(_1, _2, 1); }

__m128i vsrlrni_h_w(v8i16 _1, v8i16 _2) { return __lsx_vsrlrni_h_w(_1, _2, 1); }

__m128i vsrlrni_w_d(v4i32 _1, v4i32 _2) { return __lsx_vsrlrni_w_d(_1, _2, 1); }

__m128i vsrlrni_d_q(v2i64 _1, v2i64 _2) { return __lsx_vsrlrni_d_q(_1, _2, 1); }

__m128i vssrlni_b_h(v16i8 _1, v16i8 _2) { return __lsx_vssrlni_b_h(_1, _2, 1); }

__m128i vssrlni_h_w(v8i16 _1, v8i16 _2) { return __lsx_vssrlni_h_w(_1, _2, 1); }

__m128i vssrlni_w_d(v4i32 _1, v4i32 _2) { return __lsx_vssrlni_w_d(_1, _2, 1); }

__m128i vssrlni_d_q(v2i64 _1, v2i64 _2) { return __lsx_vssrlni_d_q(_1, _2, 1); }

__m128i vssrlni_bu_h(v16u8 _1, v16i8 _2) {
  return __lsx_vssrlni_bu_h(_1, _2, 1);
}

__m128i vssrlni_hu_w(v8u16 _1, v8i16 _2) {
  return __lsx_vssrlni_hu_w(_1, _2, 1);
}

__m128i vssrlni_wu_d(v4u32 _1, v4i32 _2) {
  return __lsx_vssrlni_wu_d(_1, _2, 1);
}

__m128i vssrlni_du_q(v2u64 _1, v2i64 _2) {
  return __lsx_vssrlni_du_q(_1, _2, 1);
}

__m128i vssrlrni_b_h(v16i8 _1, v16i8 _2) {
  return __lsx_vssrlrni_b_h(_1, _2, 1);
}

__m128i vssrlrni_h_w(v8i16 _1, v8i16 _2) {
  return __lsx_vssrlrni_h_w(_1, _2, 1);
}

__m128i vssrlrni_w_d(v4i32 _1, v4i32 _2) {
  return __lsx_vssrlrni_w_d(_1, _2, 1);
}

__m128i vssrlrni_d_q(v2i64 _1, v2i64 _2) {
  return __lsx_vssrlrni_d_q(_1, _2, 1);
}

__m128i vssrlrni_bu_h(v16u8 _1, v16i8 _2) {
  return __lsx_vssrlrni_bu_h(_1, _2, 1);
}

__m128i vssrlrni_hu_w(v8u16 _1, v8i16 _2) {
  return __lsx_vssrlrni_hu_w(_1, _2, 1);
}

__m128i vssrlrni_wu_d(v4u32 _1, v4i32 _2) {
  return __lsx_vssrlrni_wu_d(_1, _2, 1);
}

__m128i vssrlrni_du_q(v2u64 _1, v2i64 _2) {
  return __lsx_vssrlrni_du_q(_1, _2, 1);
}

__m128i vsrani_b_h(v16i8 _1, v16i8 _2) { return __lsx_vsrani_b_h(_1, _2, 1); }

__m128i vsrani_h_w(v8i16 _1, v8i16 _2) { return __lsx_vsrani_h_w(_1, _2, 1); }

__m128i vsrani_w_d(v4i32 _1, v4i32 _2) { return __lsx_vsrani_w_d(_1, _2, 1); }

__m128i vsrani_d_q(v2i64 _1, v2i64 _2) { return __lsx_vsrani_d_q(_1, _2, 1); }

__m128i vsrarni_b_h(v16i8 _1, v16i8 _2) { return __lsx_vsrarni_b_h(_1, _2, 1); }

__m128i vsrarni_h_w(v8i16 _1, v8i16 _2) { return __lsx_vsrarni_h_w(_1, _2, 1); }

__m128i vsrarni_w_d(v4i32 _1, v4i32 _2) { return __lsx_vsrarni_w_d(_1, _2, 1); }

__m128i vsrarni_d_q(v2i64 _1, v2i64 _2) { return __lsx_vsrarni_d_q(_1, _2, 1); }

__m128i vssrani_b_h(v16i8 _1, v16i8 _2) { return __lsx_vssrani_b_h(_1, _2, 1); }

__m128i vssrani_h_w(v8i16 _1, v8i16 _2) { return __lsx_vssrani_h_w(_1, _2, 1); }

__m128i vssrani_w_d(v4i32 _1, v4i32 _2) { return __lsx_vssrani_w_d(_1, _2, 1); }

__m128i vssrani_d_q(v2i64 _1, v2i64 _2) { return __lsx_vssrani_d_q(_1, _2, 1); }

__m128i vssrani_bu_h(v16u8 _1, v16i8 _2) {
  return __lsx_vssrani_bu_h(_1, _2, 1);
}

__m128i vssrani_hu_w(v8u16 _1, v8i16 _2) {
  return __lsx_vssrani_hu_w(_1, _2, 1);
}

__m128i vssrani_wu_d(v4u32 _1, v4i32 _2) {
  return __lsx_vssrani_wu_d(_1, _2, 1);
}

__m128i vssrani_du_q(v2u64 _1, v2i64 _2) {
  return __lsx_vssrani_du_q(_1, _2, 1);
}

__m128i vssrarni_b_h(v16i8 _1, v16i8 _2) {
  return __lsx_vssrarni_b_h(_1, _2, 1);
}

__m128i vssrarni_h_w(v8i16 _1, v8i16 _2) {
  return __lsx_vssrarni_h_w(_1, _2, 1);
}

__m128i vssrarni_w_d(v4i32 _1, v4i32 _2) {
  return __lsx_vssrarni_w_d(_1, _2, 1);
}

__m128i vssrarni_d_q(v2i64 _1, v2i64 _2) {
  return __lsx_vssrarni_d_q(_1, _2, 1);
}

__m128i vssrarni_bu_h(v16u8 _1, v16i8 _2) {
  return __lsx_vssrarni_bu_h(_1, _2, 1);
}

__m128i vssrarni_hu_w(v8u16 _1, v8i16 _2) {
  return __lsx_vssrarni_hu_w(_1, _2, 1);
}

__m128i vssrarni_wu_d(v4u32 _1, v4i32 _2) {
  return __lsx_vssrarni_wu_d(_1, _2, 1);
}

__m128i vssrarni_du_q(v2u64 _1, v2i64 _2) {
  return __lsx_vssrarni_du_q(_1, _2, 1);
}

__m128i vpermi_w(v4i32 _1, v4i32 _2) { return __lsx_vpermi_w(_1, _2, 1); }

__m128i vld(void *_1) { return __lsx_vld(_1, 1); }

void vst(v16i8 _1, void *_2) { return __lsx_vst(_1, _2, 1); }

__m128i vldi() { return __lsx_vldi(1); }

int bnz_b(v16u8 _1) { return __lsx_bnz_b(_1); }

int bnz_d(v2u64 _1) { return __lsx_bnz_d(_1); }

int bnz_h(v8u16 _1) { return __lsx_bnz_h(_1); }

int bnz_v(v16u8 _1) { return __lsx_bnz_v(_1); }

int bnz_w(v4u32 _1) { return __lsx_bnz_w(_1); }

int bz_b(v16u8 _1) { return __lsx_bz_b(_1); }

int bz_d(v2u64 _1) { return __lsx_bz_d(_1); }

int bz_h(v8u16 _1) { return __lsx_bz_h(_1); }

int bz_v(v16u8 _1) { return __lsx_bz_v(_1); }

int bz_w(v4u32 _1) { return __lsx_bz_w(_1); }

__m128i vrepli_b() { return __lsx_vrepli_b(1); }

__m128i vrepli_d() { return __lsx_vrepli_d(1); }

__m128i vrepli_h() { return __lsx_vrepli_h(1); }

__m128i vrepli_w() { return __lsx_vrepli_w(1); }
