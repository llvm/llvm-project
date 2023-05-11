// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s

__attribute__((target("no-paired-vector-memops")))
void test_pair(unsigned char *vqp, unsigned char *vpp, vector unsigned char vc, unsigned char *resp) {
  __vector_pair res;
  signed long offset;
  __builtin_vsx_assemble_pair(&res, vc, vc);
  __builtin_vsx_disassemble_pair(resp, (__vector_pair*)vpp);
  __vector_pair vp = __builtin_vsx_lxvp(offset, (const __vector_pair*)vpp);
  __builtin_vsx_stxvp(vp, offset, (__vector_pair*)vpp);
  __builtin_mma_xxmtacc((__vector_quad *)vpp);
  __builtin_mma_pmxvf64ger((__vector_quad *)vpp, vp, vc, 0, 0);
// CHECK: error: '__builtin_vsx_assemble_pair' needs target feature paired-vector-memops
// CHECK: error: '__builtin_vsx_disassemble_pair' needs target feature paired-vector-memops
// CHECK: error: '__builtin_vsx_lxvp' needs target feature paired-vector-memops
// CHECK: error: '__builtin_vsx_stxvp' needs target feature paired-vector-memops
// CHECK: error: '__builtin_mma_xxmtacc' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmxvf64ger' needs target feature mma,paired-vector-memops
}
