// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu pwr10 \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-cpu future \
// RUN:   %s -emit-llvm-only 2>&1 | FileCheck %s

__attribute__((target("no-mma")))
void test_mma(unsigned char *vdmrpp, unsigned char *vdmrp, unsigned char *vpp, vector unsigned char vc) {
  __dmr2048 vdmrpair = *((__dmr2048 *)vdmrpp);
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
  __builtin_mma_build_dmr(&vdmr, vc, vc, vc, vc, vc, vc, vc, vc);
  __builtin_mma_disassemble_dmr(vdmrp, &vdmr);
  __builtin_mma_dmsha2hash(&vdmr, &vdmr, 0);
  __builtin_mma_dmsha3hash(&vdmrpair, 0);
  __builtin_mma_dmxxshapad(&vdmr, vc, 0, 0, 0);

// CHECK: error: '__builtin_mma_dmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmsetdmrz' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmmr' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmxor' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_build_dmr' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_disassemble_dmr' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmsha2hash' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmsha3hash' needs target feature mma,isa-future-instructions
// CHECK: error: '__builtin_mma_dmxxshapad' needs target feature mma,isa-future-instructions
}
