// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 %s -target-feature +v -fsyntax-only -verify

#include <riscv_vector.h>

void test_zvk_features(vuint32m4_t vd, vuint32m4_t vs2, vuint32m4_t vs1, vuint64m1_t vs2_64, vuint64m1_t vs1_64, size_t vl) {
  // zvbb
  __riscv_vbrev(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvbb}}
  __riscv_vclz(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvbb}}
  __riscv_vctz(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvbb}}
  __riscv_vcpop(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvbb}}
  __riscv_vwsll(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvbb}}

  // zvbc
  __riscv_vclmul(vs2_64, vs1_64, vl); // expected-error {{builtin requires at least one of the following extensions: zvbc}}
  __riscv_vclmulh(vs2_64, vs1_64, vl); // expected-error {{builtin requires at least one of the following extensions: zvbc}}

  // zvkb
  __riscv_vandn(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvkb}}
  __riscv_vbrev8(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkb}}
  __riscv_vrev8(vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkb}}
  __riscv_vrol(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvkb}}
  __riscv_vror(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvkb}}

  // zvkg
  __riscv_vghsh(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvkg}}
  __riscv_vgmul(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvkg}}

  // zvkned
  __riscv_vaesdf_vv(vd, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaesdm_vv(vd, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaesef_vv(vd, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaesem_vv(vd, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaeskf1(vs2, 0, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaeskf2(vd, vs2, 0, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}
  __riscv_vaesz(vd, vs2, vl); // expected-error {{builtin requires at least one of the following extensions: zvkned}}

  // zvknha or zvknhb
  __riscv_vsha2ch(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknha or zvknhb}}
  __riscv_vsha2cl(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknha or zvknhb}}
  __riscv_vsha2ms(vd, vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvknha or zvknhb}}

  //zvksed
  __riscv_vsm4k(vs2, 0, vl); // expected-error {{builtin requires at least one of the following extensions: zvksed}}
  __riscv_vsm4r_vv(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvksed}}

  // zvksh
  __riscv_vsm3c(vd, vs2, 0, vl); // expected-error {{builtin requires at least one of the following extensions: zvksh}}
  __riscv_vsm3me(vs2, vs1, vl); // expected-error {{builtin requires at least one of the following extensions: zvksh}}
}
