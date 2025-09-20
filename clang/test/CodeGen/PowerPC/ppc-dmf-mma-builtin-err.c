// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature -mma \
// RUN:   -target-cpu pwr10 %s -emit-llvm-only 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,ISA_FUTURE  %s
// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature -mma \
// RUN:   -target-cpu future %s -emit-llvm-only 2>&1 | \
// RUN:   FileCheck --check-prefixes=CHECK,ISA_FUTURE  %s
// RUN: not %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature \
// RUN:   -isa-future-instructions -target-cpu future %s -emit-llvm-only 2>&1 | \
// RUN:   FileCheck --check-prefix=ISA_FUTURE %s

//__attribute__((target("no-mma")))
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
  __builtin_mma_build_dmr(&vdmr, vc, vc, vc, vc, vc, vc, vc, vc);
  __builtin_mma_disassemble_dmr(vdmrp, &vdmr);

// CHECK: error: '__builtin_mma_dmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4pp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_dmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// CHECK: error: '__builtin_mma_pmdmxvi8gerx4spp' needs target feature mma,paired-vector-memops
// ISA_FUTURE: error: '__builtin_mma_dmsetdmrz' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmmr' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxor' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_build_dmr' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_disassemble_dmr' needs target feature mma,isa-future-instructions

  // DMF VSX Vector bfloat16 GER 2x builtins.

  __builtin_mma_dmxvbf16gerx2(&vdmr, vp, vc);
  __builtin_mma_dmxvbf16gerx2nn(&vdmr, vp, vc);
  __builtin_mma_dmxvbf16gerx2np(&vdmr, vp, vc);
  __builtin_mma_dmxvbf16gerx2pn(&vdmr, vp, vc);
  __builtin_mma_dmxvbf16gerx2pp(&vdmr, vp, vc);
  __builtin_mma_pmdmxvbf16gerx2(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvbf16gerx2nn(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvbf16gerx2np(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvbf16gerx2pn(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvbf16gerx2pp(&vdmr, vp, vc, 0, 0, 0);

// ISA_FUTURE: error: '__builtin_mma_dmxvbf16gerx2' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvbf16gerx2nn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvbf16gerx2np' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvbf16gerx2pn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvbf16gerx2pp' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvbf16gerx2' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvbf16gerx2nn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvbf16gerx2np' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvbf16gerx2pn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvbf16gerx2pp' needs target feature mma,isa-future-instructions

  // DMF VSX Vector 16-bitFloating-point GER 2x builtins.

  __builtin_mma_dmxvf16gerx2(&vdmr, vp, vc);
  __builtin_mma_dmxvf16gerx2nn(&vdmr, vp, vc);
  __builtin_mma_dmxvf16gerx2np(&vdmr, vp, vc);
  __builtin_mma_dmxvf16gerx2pn(&vdmr, vp, vc);
  __builtin_mma_dmxvf16gerx2pp(&vdmr, vp, vc);
  __builtin_mma_pmdmxvf16gerx2(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvf16gerx2nn(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvf16gerx2np(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvf16gerx2pn(&vdmr, vp, vc, 0, 0, 0);
  __builtin_mma_pmdmxvf16gerx2pp(&vdmr, vp, vc, 0, 0, 0);

// ISA_FUTURE: error: '__builtin_mma_dmxvf16gerx2' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvf16gerx2nn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvf16gerx2np' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvf16gerx2pn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_dmxvf16gerx2pp' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvf16gerx2' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvf16gerx2nn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvf16gerx2np' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvf16gerx2pn' needs target feature mma,isa-future-instructions
// ISA_FUTURE: error: '__builtin_mma_pmdmxvf16gerx2pp' needs target feature mma,isa-future-instructions
}
