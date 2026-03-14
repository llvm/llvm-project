// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s

__attribute__((target("no-mma")))
void test_mma(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_quad vq = *((__vector_quad *)vqp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_xxmtacc(&vq);
  *((__vector_quad *)resp) = vq;
  __builtin_mma_pmxvf64ger(&vq, vp, vc, 0, 0);
// CHECK: error: '__builtin_mma_xxmtacc' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmxvf64ger' needs target feature mma,paired-vector-memops
}
