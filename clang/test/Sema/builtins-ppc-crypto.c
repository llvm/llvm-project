// RUN: %clang_cc1 -triple powerpc64-unknown-unknown -target-cpu future \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu future \
// RUN:   -fsyntax-only -verify %s


void test_crypto(unsigned char *vdmrpp, unsigned char *vdmrp, unsigned char *vpp, vector unsigned char vc) {
  __dmr2048 vdmrpair = *((__dmr2048 *)vdmrpp);
  __dmr1024 vdmr = *((__dmr1024 *)vdmrp);
  __vector_pair vp = *((__vector_pair *)vpp);
  int ia;

  __builtin_dmsha2hash(&vdmr, &vdmr, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  __builtin_dmsha2hash(&vdmr, &vdmr, -1);  // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  __builtin_dmsha2hash(&vdmr, &vdmr, ia);  // expected-error {{argument to '__builtin_dmsha2hash' must be a constant integer}}

  __builtin_dmsha3hash(&vdmrpair, 32);  // expected-error {{argument value 32 is outside the valid range [0, 31]}}
  __builtin_dmsha3hash(&vdmrpair, -2);  // expected-error {{argument value -2 is outside the valid range [0, 31]}}
  __builtin_dmsha3hash(&vdmrpair, ia);  // expected-error {{argument to '__builtin_dmsha3hash' must be a constant integer}}

  __builtin_dmxxshapad(&vdmr, vc, 4, 0, 3);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  __builtin_dmxxshapad(&vdmr, vc, 3, 2, 3);  // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  __builtin_dmxxshapad(&vdmr, vc, 3, 1, -1);  // expected-error {{argument value -1 is outside the valid range [0, 3]}}
  __builtin_dmxxshapad(&vdmr, vc, ia, 1, 1);  // expected-error {{argument to '__builtin_dmxxshapad' must be a constant integer}}
  __builtin_dmxxshapad(&vdmr, vc, 0, ia, 1);  // expected-error {{argument to '__builtin_dmxxshapad' must be a constant integer}}
  __builtin_dmxxshapad(&vdmr, vc, 0, 1, ia);  // expected-error {{argument to '__builtin_dmxxshapad' must be a constant integer}}

  __builtin_dmxxsha3512pad(&vdmr, vc, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  __builtin_dmxxsha3512pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxsha3512pad' must be a constant integer}}

  __builtin_dmxxsha3384pad(&vdmr, vc, 3);  // expected-error {{argument value 3 is outside the valid range [0, 1]}}
  __builtin_dmxxsha3384pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxsha3384pad' must be a constant integer}}

  __builtin_dmxxsha3256pad(&vdmr, vc, -1);  // expected-error {{argument value -1 is outside the valid range [0, 1]}}
  __builtin_dmxxsha3256pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxsha3256pad' must be a constant integer}}

  __builtin_dmxxsha3224pad(&vdmr, vc, 4);  // expected-error {{argument value 4 is outside the valid range [0, 1]}}
  __builtin_dmxxsha3224pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxsha3224pad' must be a constant integer}}

  __builtin_dmxxshake256pad(&vdmr, vc, -2);  // expected-error {{argument value -2 is outside the valid range [0, 1]}}
  __builtin_dmxxshake256pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxshake256pad' must be a constant integer}}

  __builtin_dmxxshake128pad(&vdmr, vc, 2);  // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  __builtin_dmxxshake128pad(&vdmr, vc, ia);  // expected-error {{argument to '__builtin_dmxxshake128pad' must be a constant integer}}
}
