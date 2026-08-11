// Straight-line selection: vadd lowers to the prologue, gid computation,
// A64 loads, a store, and EOT.
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' | FileCheck %s
// RUN: inter-opt %s --pass-pipeline='builtin.module(transform-preload-library{transform-library-paths=%inter_pipelines},transform-interpreter{entry-point=inter_backend})' | inter-translate --xemachine-to-ged -o %t
// RUN: inter-ged-dump %t | FileCheck %s --check-prefix=GED

module {
  // CHECK: func.func @vadd
  // CHECK-SAME: xemachine.barrier_count = 0 : i32
  // CHECK-SAME: xemachine.grf_count = 128 : i32
  // CHECK-SAME: xemachine.grf_used = 28 : i32
  // CHECK-SAME: xemachine.has_global_atomics = false
  // CHECK-SAME: xemachine.has_no_stateless_write = false
  // CHECK-SAME: xemachine.inline_data_payload_size = 32 : i32
  // CHECK-SAME: xemachine.kernel_type = (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
  // CHECK-SAME: xemachine.payload_entry_offset = 192 : i32
  // CHECK-SAME: xemachine.per_thread_payload_size = 192 : i32
  // CHECK-SAME: xemachine.reserved_grf_count = 5 : i32
  // CHECK-SAME: xemachine.simd_size = 32 : i32
  // CHECK-SAME: xemachine.target = #xemachine.target<chip = "bmg">
  // CHECK-SAME: xemachine.uses_thread_ids
  llvm.func spir_kernelcc @vadd(%a: !llvm.ptr<1>, %b: !llvm.ptr<1>,
                                %out: !llvm.ptr<1>) {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %gid = llvm.call spir_funccc @_Z13get_global_idj(%c0) : (i32) -> i64
    %pa = llvm.getelementptr %a[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %va = llvm.load %pa : !llvm.ptr<1> -> i32
    %pb = llvm.getelementptr %b[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    %vb = llvm.load %pb : !llvm.ptr<1> -> i32
    %sum = llvm.add %vb, %va : i32
    %po = llvm.getelementptr %out[%gid] : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    llvm.store %sum, %po : i32, !llvm.ptr<1>
    llvm.return
  }
  llvm.func spir_funccc @_Z13get_global_idj(i32) -> i64
}

// Prologue: blob base and the two payload loads.
// Ready immediates are pulled ahead of real instructions by the scheduler.
// CHECK: [[PAYLOAD_STRIDE:%.*]] = xemachine.imm 192
// CHECK: xemachine.and
// CHECK: [[SLOT:%.*]] = xemachine.and {{.*}}src0Sub = 4
// CHECK: [[THREAD_OFFSET_ACC:%.*]] = xemachine.mul [[SLOT]], [[PAYLOAD_STRIDE]]
// CHECK: xemachine.load_block_a32
// CHECK: [[THREAD_OFFSET:%.*]] = xemachine.mov [[THREAD_OFFSET_ACC]]
// CHECK: xemachine.add {{.*}}, [[THREAD_OFFSET]]
// gid work can fill the gap between the two payload loads.
// CHECK: xemachine.mul
// CHECK: xemachine.load_block_a32
// CHECK: xemachine.add3
// The named backend pipeline prepares destructive updates before scheduling.
// CHECK: [[UPDATE_BASE:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-base"
// CHECK-NEXT: [[UPDATE_VALUE:%.*]] = xemachine.mov {{.*}}xemachine.regalloc_copy = "update-value"
// CHECK-NEXT: xemachine.update_tuple [[UPDATE_BASE]], [[UPDATE_VALUE]]
// Two A64 loads and one store, each with an explicit four-GRF address tuple.
// CHECK: [[ADDR0:%.*]] = xemachine.tuple_from_elements
// CHECK-SAME: -> !xemachine.reg<64,
// CHECK-NEXT: xemachine.load_a64 [[ADDR0]]
// CHECK: [[ADDR1:%.*]] = xemachine.tuple_from_elements
// CHECK-SAME: -> !xemachine.reg<64,
// CHECK-NEXT: xemachine.load_a64 [[ADDR1]]
// CHECK: [[ADDR2:%.*]] = xemachine.tuple_from_elements
// CHECK-SAME: -> !xemachine.reg<64,
// CHECK-NEXT: xemachine.store_a64 [[ADDR2]] {{.*}}data
// CHECK: [[FINAL:%.*]] = xemachine.token_join
// CHECK: xemachine.sync allrd
// EOT via the gateway.
// CHECK-NEXT: xemachine.eot {{.*}} dep [[FINAL]]

// GED: pc=144 opcode=sync {{.*}}function=allwr
// GED: opcode=mul
// GED: opcode=add3
// GED: opcode=shl
// GED: pc=464 opcode=add
// GED-NEXT: pc=480 opcode=add
// GED-NEXT: pc=496 opcode=send exec=32 swsb=0x322
// GED: opcode=send {{.*}}sfid=ugm {{.*}}len=2 eot=0
// GED: opcode=send {{.*}}sfid=gateway {{.*}}len=0 eot=1
