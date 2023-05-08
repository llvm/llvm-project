// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 \
// RUN:   -target-feature -paired-vector-memops -fsyntax-only %s -verify
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr9 \
// RUN:   -fsyntax-only %s -verify

void test1(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_pair res;
  __builtin_vsx_assemble_pair(&res, vc, vc); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
}

void test2(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __builtin_vsx_disassemble_pair(resp, (__vector_pair*)vpp); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
}

void test3(const __vector_pair *vpp, signed long long offset, const __vector_pair *vp2) {
  __vector_pair vp = __builtin_vsx_lxvp(offset, vpp); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
  __builtin_vsx_stxvp(vp, offset, vp2); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
}

void test4(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_quad vq = *((__vector_quad *)vqp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_xxmtacc(&vq); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
  *((__vector_quad *)resp) = vq;
}


