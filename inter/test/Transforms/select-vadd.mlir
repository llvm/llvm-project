// Straight-line selection: vadd lowers to the prologue, gid computation,
// A64 loads, a store, and EOT.
// RUN: inter-opt %s --inter-normalize-cf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync | FileCheck %s
// RUN: inter-opt %s --inter-normalize-cf --inter-convert-calls --inter-convert-memory --inter-select-to-machine --inter-insert-sync | inter-translate --xemachine-to-ged -o %t
// RUN: inter-ged-dump %t | FileCheck %s --check-prefix=GED

module {
  // CHECK: func.func @vadd
  // CHECK-SAME: xemachine.grf_count = 128 : i32
  // CHECK-SAME: xemachine.inline_data_payload_size = 32 : i32
  // CHECK-SAME: xemachine.kernel_type = (!llvm.ptr<1>, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
  // CHECK-SAME: xemachine.payload_entry_offset = 192 : i32
  // CHECK-SAME: xemachine.per_thread_payload_size = 192 : i32
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
// CHECK: xemachine.and
// CHECK: [[SLOT:%.*]] = xemachine.and {{.*}}src0Sub = 4
// CHECK: [[PAYLOAD_STRIDE:%.*]] = xemachine.imm 192
// CHECK: [[THREAD_OFFSET_ACC:%.*]] = xemachine.mul [[SLOT]], [[PAYLOAD_STRIDE]]
// CHECK: [[THREAD_OFFSET:%.*]] = xemachine.mov [[THREAD_OFFSET_ACC]]
// CHECK: xemachine.add {{.*}}, [[THREAD_OFFSET]]
// CHECK-COUNT-2: xemachine.load_block_a32
// gid: mul into acc, then add the hardware- or software-provided local ID.
// CHECK: xemachine.mul
// CHECK: xemachine.add3
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

// GED: pc=192 opcode=sync {{.*}}function=allwr
// GED: opcode=mul
// GED: opcode=add3
// GED: opcode=shl
// GED: pc=384 opcode=add
// GED-NEXT: pc=400 opcode=add
// GED-NEXT: pc=416 opcode=send exec=32 swsb=0x322
// GED: opcode=send {{.*}}sfid=ugm {{.*}}len=2 eot=0
// GED: opcode=send {{.*}}sfid=gateway {{.*}}len=0 eot=1
