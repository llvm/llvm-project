// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu future \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s

__attribute__((target("no-paired-vector-memops")))
void test_pair(unsigned char *vdmr, unsigned char *vpp, vector unsigned char vc) {
   __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_dmxvi8gerx4((__dmr1024 *)vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4((__dmr1024 *)vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_dmxvi8gerx4pp((__dmr1024 *)vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4pp((__dmr1024 *)vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_dmxvi8gerx4spp((__dmr1024 *)vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4spp((__dmr1024 *)vdmr, vp, vc, 0, 0, 0);

// CHECK: error: '__builtin_mma_dmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4spp' needs target feature mma,paired-vector-memops
}
