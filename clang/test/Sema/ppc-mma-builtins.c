// REQUIRES: powerpc-registered-target
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 \
// RUN:   -target-feature -mma -fsyntax-only %s -verify

void test1(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_pair res;
  __builtin_vsx_assemble_pair(&res, vc, vc);
}

void test2(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __builtin_vsx_disassemble_pair(resp, (__vector_pair*)vpp);
}

void test3(const __vector_pair *vpp, signed long offset, __vector_pair *vp2) {
  __vector_pair vp = __builtin_vsx_lxvp(offset, vpp);
  __builtin_vsx_stxvp(vp, offset, vp2);
}

void test4(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_quad vq = *((__vector_quad *)vqp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_xxmtacc(&vq); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
  *((__vector_quad *)resp) = vq;
}

void test5(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_quad vq = *((__vector_quad *)vqp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_pmxvf64ger(&vq, vp, vc, 0, 0); // expected-error {{this builtin is only valid on POWER10 or later CPUs}}
  *((__vector_quad *)resp) = vq;
}


