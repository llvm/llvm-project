// RUN: inter-opt %s --inter-normalize-pointers | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<
    #dlti.dl_entry<!llvm.ptr<1>, dense<[64, 64, 64, 32]> : vector<4xi64>>,
    #dlti.dl_entry<!llvm.ptr<3>, dense<[32, 32, 32, 32]> : vector<4xi64>>
  >} {
  // CHECK-LABEL: func.func @dynamic_index
  // CHECK: [[INDEX:%.*]] = llvm.trunc %{{.*}} : i64 to i32
  // CHECK: [[SCALE:%.*]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: [[OFFSET:%.*]] = llvm.mul [[INDEX]], [[SCALE]] : i32
  // CHECK: [[PTR:%.*]] = xw.ptradd %{{.*}}, [[OFFSET]] : !llvm.ptr<1>, i32
  // CHECK-NOT: llvm.getelementptr
  func.func @dynamic_index(%base: !llvm.ptr<1>, %index: i64) attributes {
      xemachine.kernel} {
    %ptr = llvm.getelementptr %base[%index]
        : (!llvm.ptr<1>, i64) -> !llvm.ptr<1>, i32
    return
  }

  // An unpacked {i8, i32, [3 x i16]} places the array at byte eight.
  // CHECK-LABEL: func.func @aggregate
  // CHECK: [[EIGHT:%.*]] = llvm.mlir.constant(8 : i32) : i32
  // CHECK: [[TWO:%.*]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[DYNAMIC:%.*]] = llvm.mul %{{.*}}, [[TWO]] overflow<nsw> : i32
  // CHECK: [[OFFSET:%.*]] = llvm.add [[EIGHT]], [[DYNAMIC]] overflow<nsw> : i32
  // CHECK: xw.ptradd %{{.*}}, [[OFFSET]] {gep_flags = 3 : i32}
  // CHECK-NOT: llvm.getelementptr
  func.func @aggregate(%base: !llvm.ptr<3>, %index: i32) attributes {
      xemachine.kernel} {
    %ptr = llvm.getelementptr inbounds %base[0, 2, %index]
        : (!llvm.ptr<3>, i32) -> !llvm.ptr<3>,
          !llvm.struct<(i8, i32, array<3 x i16>)>
    return
  }

  // CHECK-LABEL: func.func @chained
  // CHECK-COUNT-2: xw.ptradd
  // CHECK-NOT: llvm.getelementptr
  func.func @chained(%base: !llvm.ptr<1>, %first: i32, %second: i32)
      attributes {xemachine.kernel} {
    %ptr0 = llvm.getelementptr %base[%first]
        : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i16
    %ptr1 = llvm.getelementptr %ptr0[%second]
        : (!llvm.ptr<1>, i32) -> !llvm.ptr<1>, i8
    return
  }
}
