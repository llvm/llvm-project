// RUN: inter-opt %s | FileCheck %s
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_regalloc})' | FileCheck %s --check-prefix=ALLOC

func.func @subgrf_update() attributes {
    xemachine.grf_count = 16 : i32,
    xemachine.reserved_grf_count = 0 : i32} {
  %zero = xemachine.imm 0 : i32
  %base = xemachine.mov %zero {execSize = 32 : i32, noMask}
      : (!xemachine.imm, i32) -> !xemachine.reg<32, -1>
  %update = xemachine.mov %zero {execSize = 16 : i32, noMask}
      : (!xemachine.imm, i32) -> !xemachine.reg<16, -1>
  %result = xemachine.update_tuple %base, %update {offsets = [8]}
      : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
      -> !xemachine.reg<32, -1>
  return
}

// CHECK: xemachine.update_tuple {{.*}} {offsets = [8]}

// ALLOC: [[UPDATE:%.*]] = xemachine.mov {{.*}}dstSub = 8 : i32{{.*}}!xemachine.reg<16, [[BASE:[0-9]+]]>
// ALLOC-NEXT: xemachine.update_tuple {{.*}}, [[UPDATE]] {offsets = [8]} {{.*}}!xemachine.reg<32, [[BASE]]>
