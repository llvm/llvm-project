// REQUIRES: powerpc-registered-target

// RUN: %clang_cc1 -target-feature +altivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -fsyntax-only   \
// RUN: -flax-vector-conversions=integer \
// RUN: -Wall -Werror -verify %s

// RUN: %clang_cc1 -target-feature +altivec -target-feature +power9-vector  \
// RUN: -triple powerpc64le-unknown-unknown -fsyntax-only    \
// RUN: -flax-vector-conversions=integer \
// RUN: -Wall -Werror -verify %s

// FIXME: Fix <altivec.h> so this test also passes under
// -flax-vector-conversions=none (this last test exists to produce an error if
// we change the default to that without fixing <altivec.h>).
// RUN: %clang_cc1 -target-feature +altivec -target-feature +power9-vector \
// RUN:   -triple powerpc64-unknown-unknown -fsyntax-only   \
// RUN: -Wall -Werror -verify %s

#include <altivec.h>

extern vector signed int vsi;
extern vector signed int vui;
extern vector float vf;
extern vector unsigned char vuc;
extern vector signed __int128 vsllli;

void testInsertWord(void) {
  int index = 5;
  vector unsigned char v1 = vec_insert4b(vsi, vuc, index); // expected-error {{argument to '__builtin_vsx_insertword' must be a constant integer}}
  vector unsigned long long v2 = vec_extract4b(vuc, index);   // expected-error {{argument to '__builtin_vsx_extractuword' must be a constant integer}}
}

void testXXPERMDI(int index) {
  vec_xxpermdi(vsi); //expected-error {{too few arguments to function call, expected 3, have 1}}
  vec_xxpermdi(vsi, vsi, 2, 4); //expected-error {{too many arguments to function call, expected 3, have 4}}
  vec_xxpermdi(vsi, vsi, index); //expected-error {{argument 3 to '__builtin_vsx_xxpermdi' must be a 2-bit unsigned literal (i.e. 0, 1, 2 or 3)}}
  vec_xxpermdi(1, 2, 3); //expected-error {{first two arguments to '__builtin_vsx_xxpermdi' must be vectors}}
  vec_xxpermdi(vsi, vuc, 2); //expected-error {{first two arguments to '__builtin_vsx_xxpermdi' must have the same type}}
}

void testXXSLDWI(int index) {
  vec_xxsldwi(vsi); //expected-error {{too few arguments to function call, expected 3, have 1}}
  vec_xxsldwi(vsi, vsi, 2, 4); //expected-error {{too many arguments to function call, expected 3, have 4}}
  vec_xxsldwi(vsi, vsi, index); //expected-error {{argument 3 to '__builtin_vsx_xxsldwi' must be a 2-bit unsigned literal (i.e. 0, 1, 2 or 3)}}
  vec_xxsldwi(1, 2, 3); //expected-error {{first two arguments to '__builtin_vsx_xxsldwi' must be vectors}}
  vec_xxsldwi(vsi, vuc, 2); //expected-error {{first two arguments to '__builtin_vsx_xxsldwi' must have the same type}}
}

void testCTF(int index) {
  vec_ctf(vsi, index); //expected-error {{argument to '__builtin_altivec_vcfsx' must be a constant integer}}
  vec_ctf(vui, index); //expected-error {{argument to '__builtin_altivec_vcfsx' must be a constant integer}}
}

void testVCFSX(int index) {
  vec_vcfsx(vsi, index); //expected-error {{argument to '__builtin_altivec_vcfsx' must be a constant integer}}
}

void testVCFUX(int index) {
  vec_vcfux(vui, index); //expected-error {{argument to '__builtin_altivec_vcfux' must be a constant integer}}
}

void testCTS(int index) {
  vec_cts(vf, index); //expected-error {{argument to '__builtin_altivec_vctsxs' must be a constant integer}}

}

void testVCTSXS(int index) {
  vec_vctsxs(vf, index); //expected-error {{argument to '__builtin_altivec_vctsxs' must be a constant integer}}
}

void testCTU(int index) {
  vec_ctu(vf, index); //expected-error {{argument to '__builtin_altivec_vctuxs' must be a constant integer}}

}

void testVCTUXS(int index) {
  vec_vctuxs(vf, index); //expected-error {{argument to '__builtin_altivec_vctuxs' must be a constant integer}}
}

void testUnpack128(int index) {
  __builtin_unpack_vector_int128(vsllli, index); //expected-error {{argument to '__builtin_unpack_vector_int128' must be a constant integer}}
  __builtin_unpack_vector_int128(vsllli, 5); //expected-error {{argument value 5 is outside the valid range [0, 1]}}
}

void testDSS(int index) {
  vec_dss(index); //expected-error {{argument to '__builtin_altivec_dss' must be a constant integer}}
  vec_dss(5); //expected-error {{argument value 5 is outside the valid range [0, 3]}}
}

void testDST(int index) {
  vec_dst(&vsi, index, index); //expected-error {{argument to '__builtin_altivec_dst' must be a constant integer}}
  vec_dst(&vsi, index, 5); //expected-error {{argument value 5 is outside the valid range [0, 3]}}
  vec_dstt(&vsi, index, index); //expected-error {{argument to '__builtin_altivec_dstt' must be a constant integer}}
  vec_dstt(&vsi, index, 5); //expected-error {{argument value 5 is outside the valid range [0, 3]}}
  vec_dstst(&vsi, index, index); //expected-error {{argument to '__builtin_altivec_dstst' must be a constant integer}}
  vec_dstst(&vsi, index, 5); //expected-error {{argument value 5 is outside the valid range [0, 3]}}
  vec_dststt(&vsi, index, index); //expected-error {{argument to '__builtin_altivec_dststt' must be a constant integer}}
  vec_dststt(&vsi, index, 5); //expected-error {{argument value 5 is outside the valid range [0, 3]}}
}
