// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature -mma \
// RUN:   -target-cpu pwr10 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-linux-gnu -target-feature -mma \
// RUN:   -target-cpu future -fsyntax-only -verify %s

void test_mma(unsigned char *vdmrpp, unsigned char *vdmrp, unsigned char *vpp, vector unsigned char vc) {
  __dmr2048 vdmrpair = *((__dmr2048 *)vdmrpp);
  __dmr1024 vdmr = *((__dmr1024 *)vdmrp);
  __vector_pair vp = *((__vector_pair *)vpp);
  __builtin_mma_dmxvi8gerx4(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvi8gerx4' needs target feature mma,paired-vector-memops}}
  __builtin_mma_pmdmxvi8gerx4(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvi8gerx4' needs target feature mma,paired-vector-memops}}
  __builtin_mma_dmxvi8gerx4pp(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvi8gerx4pp' needs target feature mma,paired-vector-memops}}
  __builtin_mma_pmdmxvi8gerx4pp(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvi8gerx4pp' needs target feature mma,paired-vector-memops}}
  __builtin_mma_dmxvi8gerx4spp(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvi8gerx4spp' needs target feature mma,paired-vector-memops}}
  __builtin_mma_pmdmxvi8gerx4spp(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvi8gerx4spp' needs target feature mma,paired-vector-memops}}
  __builtin_dmsetdmrz(&vdmr); // expected-error {{'__builtin_dmsetdmrz' needs target feature mma,isa-future-instructions}}
  __builtin_dmmr(&vdmr, (__dmr1024*)vpp); // expected-error {{'__builtin_dmmr' needs target feature mma,isa-future-instructions}}
  __builtin_dmxor(&vdmr, (__dmr1024*)vpp); // expected-error {{'__builtin_dmxor' needs target feature mma,isa-future-instructions}}
  __builtin_build_dmr(&vdmr, vc, vc, vc, vc, vc, vc, vc, vc); // expected-error {{'__builtin_build_dmr' needs target feature mma,isa-future-instructions}}
  __builtin_disassemble_dmr(vdmrp, &vdmr); // expected-error {{'__builtin_disassemble_dmr' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmsha2hash(&vdmr, &vdmr, 0); // expected-error {{'__builtin_mma_dmsha2hash' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmsha3hash(&vdmrpair, 0); // expected-error {{'__builtin_mma_dmsha3hash' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxxshapad(&vdmr, vc, 0, 0, 0); // expected-error {{'__builtin_mma_dmxxshapad' needs target feature mma,isa-future-instructions}}

  // DMF VSX Vector bfloat16 GER 2x builtins.

  __builtin_mma_dmxvbf16gerx2(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvbf16gerx2' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvbf16gerx2nn(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvbf16gerx2nn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvbf16gerx2np(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvbf16gerx2np' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvbf16gerx2pn(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvbf16gerx2pn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvbf16gerx2pp(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvbf16gerx2pp' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvbf16gerx2(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvbf16gerx2' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvbf16gerx2nn(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvbf16gerx2nn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvbf16gerx2np(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvbf16gerx2np' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvbf16gerx2pn(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvbf16gerx2pn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvbf16gerx2pp(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvbf16gerx2pp' needs target feature mma,isa-future-instructions}}

  // DMF VSX Vector 16-bitFloating-point GER 2x builtins.

  __builtin_mma_dmxvf16gerx2(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvf16gerx2' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvf16gerx2nn(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvf16gerx2nn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvf16gerx2np(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvf16gerx2np' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvf16gerx2pn(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvf16gerx2pn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_dmxvf16gerx2pp(&vdmr, vp, vc); // expected-error {{'__builtin_mma_dmxvf16gerx2pp' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvf16gerx2(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvf16gerx2' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvf16gerx2nn(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvf16gerx2nn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvf16gerx2np(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvf16gerx2np' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvf16gerx2pn(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvf16gerx2pn' needs target feature mma,isa-future-instructions}}
  __builtin_mma_pmdmxvf16gerx2pp(&vdmr, vp, vc, 0, 0, 0); // expected-error {{'__builtin_mma_pmdmxvf16gerx2pp' needs target feature mma,isa-future-instructions}}
}
