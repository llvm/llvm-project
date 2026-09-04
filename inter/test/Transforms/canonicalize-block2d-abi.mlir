// RUN: inter-opt %s --inter-canonicalize-block2d-abi | FileCheck %s
// RUN: inter-opt %s --inter-canonicalize-block2d-abi \
// RUN:   --inter-convert-llvm-to-xw | FileCheck %s --check-prefix=XW

module {
  llvm.func @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(!llvm.ptr<1>, i32, i32, i32, vector<2xi32>)
  llvm.func @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr)
  llvm.func @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr)
  llvm.func @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr)

  func.func @block2d(%base: !llvm.ptr<1>) attributes {
      xw.kernel,
      xw.kernel_args = [{access = "read_write", address_space = 1 : i32, alignment = 8 : i64, kind = "pointer", offset = 24 : i64, size = 8 : i64}],
      xw.simd_width = 16 : i32} {
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %c1 = llvm.mlir.constant(1 : i32) : i32
    %c8 = llvm.mlir.constant(8 : i32) : i32
    %c128 = llvm.mlir.constant(128 : i32) : i32
    %zero = llvm.mlir.constant(dense<0> : vector<2xi32>) : vector<2xi32>
    %x = llvm.insertelement %c0, %zero[%c0 : i32] : vector<2xi32>
    %xy = llvm.insertelement %c8, %x[%c1 : i32] : vector<2xi32>
    llvm.call @_Z45intel_sub_group_2d_block_prefetch_16b_8r16x1cPU3AS1viiiDv2_i(%base, %c128, %c128, %c128, %xy) {xw.cache_control = {l1 = #xw.cache_policy<cached>, l3 = #xw.cache_policy<cached>}} : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>) -> ()

    %read_buffer = llvm.alloca %c8 x i16 : (i32) -> !llvm.ptr
    llvm.call @_Z41intel_sub_group_2d_block_read_16b_8r16x1cPU3AS1viiiDv2_iPt(%base, %c128, %c128, %c128, %xy, %read_buffer) : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    %read = llvm.load %read_buffer : !llvm.ptr -> vector<8xi16>

    %transform_buffer = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
    llvm.call @_Z52intel_sub_group_2d_block_read_transform_16b_16r16x1cPU3AS1viiiDv2_iPj(%base, %c128, %c128, %c128, %xy, %transform_buffer) : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    %transformed = llvm.load %transform_buffer : !llvm.ptr -> vector<8xi32>

    %write_buffer = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
    llvm.store %transformed, %write_buffer : vector<8xi32>, !llvm.ptr
    llvm.call @_Z42intel_sub_group_2d_block_write_32b_8r16x1cPU3AS1viiiDv2_iPj(%base, %c128, %c128, %c128, %xy, %write_buffer) : (!llvm.ptr<1>, i32, i32, i32, vector<2xi32>, !llvm.ptr) -> ()
    return
  }
}

// CHECK-LABEL: func.func @block2d
// CHECK: xw.block2d_prefetch
// CHECK-SAME: xw.cache_control
// CHECK: xw.block2d_read
// CHECK: xw.block2d_read
// CHECK-SAME: vnni
// CHECK: xw.block2d_write
// CHECK-NOT: llvm.alloca
// CHECK-NOT: llvm.call

// XW-LABEL: func.func @block2d
// XW: xw.block2d_prefetch
// XW-COUNT-2: xw.block2d_read
// XW: xw.block2d_write
// XW-NOT: unrealized_conversion_cast
// XW-NOT: llvm
