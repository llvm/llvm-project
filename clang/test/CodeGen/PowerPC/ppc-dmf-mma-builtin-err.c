// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu future \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s

__attribute__((target("no-mma")))
void test_mma(unsigned char *vdmrp, unsigned char *vpp, vector unsigned char vc) {
  __dmr1024 vdmr = *((__dmr1024 *)vdmrp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_dmxvi8gerx4(&vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_dmxvi8gerx4pp(&vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4pp(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_dmxvi8gerx4spp(&vdmr, vp, vc);
  __builtin_mma_pmdmxvi8gerx4spp(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_dmsetdmrz(&vdmr);
  __builtin_mma_dmmr(&vdmr, (__dmr1024*)vpp);
  __builtin_mma_dmxor(&vdmr, (__dmr1024*)vpp);

// CHECK: error: '__builtin_mma_dmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmsetdmrz' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmmr' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmxor' needs target feature mma,isa-future-instructions
}
