// Frozen from llvm/lighthouse at ec3a77574cc5f049736f47b121bdd4aeeb854201.
// Generated with:
//   examples/xegpu/matmul.py --sizes 128 128 64 --wg-tile 64 64
//     --dump-kernel=xegpu-wg --no-accumulate-c

module attributes {gpu.container_module} {
  func.func private @rtclock() -> f64
  func.func @__benchmark(%arg0: memref<128x128xf32>, %arg1: memref<128x64xf16>, %arg2: memref<64x128xf16>, %arg3: memref<?xf64>, %arg4: index, %arg5: index) attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    scf.for %arg6 = %c0 to %arg5 step %c1 {
      func.call @payload(%arg0, %arg1, %arg2) : (memref<128x128xf32>, memref<128x64xf16>, memref<64x128xf16>) -> ()
    }
    scf.for %arg6 = %c0 to %arg4 step %c1 {
      %0 = func.call @rtclock() : () -> f64
      func.call @payload(%arg0, %arg1, %arg2) : (memref<128x128xf32>, memref<128x64xf16>, memref<64x128xf16>) -> ()
      %1 = func.call @rtclock() : () -> f64
      %2 = arith.subf %1, %0 : f64
      memref.store %2, %arg3[%arg6] : memref<?xf64>
    }
    return
  }
  func.func @payload(%arg0: memref<128x128xf32>, %arg1: memref<128x64xf16>, %arg2: memref<64x128xf16>) attributes {llvm.emit_c_interface} {
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c256 = arith.constant 256 : index
    gpu.launch_func @payload_kernel::@payload_kernel blocks in (%c2, %c2, %c1) threads in (%c256, %c1, %c1) args(%arg1 : memref<128x64xf16>, %arg2 : memref<64x128xf16>, %arg0 : memref<128x128xf32>)
    return
  }
  gpu.module @payload_kernel [#xevm.target<O = 3>] {
    gpu.func @payload_kernel(%arg0: memref<128x64xf16>, %arg1: memref<64x128xf16>, %arg2: memref<128x128xf32>) kernel attributes {known_block_size = array<i32: 256, 1, 1>, known_grid_size = array<i32: 2, 2, 1>} {
      %cst = arith.constant dense<0.000000e+00> : vector<64x64xf32>
      %c32 = arith.constant 32 : index
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = arith.muli %block_id_x, %c64 overflow<nsw> : index
      %1 = arith.muli %block_id_y, %c64 overflow<nsw> : index
      %2 = xegpu.create_nd_tdesc %arg0 : memref<128x64xf16> -> !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %2[%0, %c0] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [8, 2], sg_data = [8, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %3 = xegpu.create_nd_tdesc %arg1 : memref<64x128xf16> -> !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.prefetch_nd %3[%c0, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [8, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
      %4 = scf.for %arg3 = %c0 to %c64 step %c32 iter_args(%arg4 = %cst) -> (vector<64x64xf32>) {
        %6 = arith.addi %arg3, %c32 : index
        xegpu.prefetch_nd %3[%6, %1] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [8, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.prefetch_nd %2[%0, %6] <{l1_hint = #xegpu.cache_hint<cached>, l2_hint = #xegpu.cache_hint<cached>, l3_hint = #xegpu.cache_hint<cached>, layout = #xegpu.layout<sg_layout = [8, 2], sg_data = [8, 16], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %7 = xegpu.load_nd %2[%0, %arg3] <{layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [16, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<64x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<64x32xf16>
        %8 = xegpu.load_nd %3[%arg3, %1] <{layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x64xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x64xf16>
        %9 = xegpu.dpas %7, %8, %arg4 {layout_a = #xegpu.layout<sg_layout = [4, 4], sg_data = [16, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [4, 4], sg_data = [32, 16], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [4, 4], sg_data = [16, 16], inst_data = [8, 16]>} : vector<64x32xf16>, vector<32x64xf16>, vector<64x64xf32> -> vector<64x64xf32>
        scf.yield %9 : vector<64x64xf32>
      }
      %5 = xegpu.create_nd_tdesc %arg2 : memref<128x128xf32> -> !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      xegpu.store_nd %4, %5[%0, %1] <{layout = #xegpu.layout<sg_layout = [4, 4], sg_data = [16, 16], inst_data = [8, 16]>}> : vector<64x64xf32>, !xegpu.tensor_desc<64x64xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
      gpu.return
    }
  }
  func.func @gpu_alloc_2d_f32(%arg0: i32, %arg1: i32) -> memref<?x?xf32> attributes {llvm.emit_c_interface} {
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %memref = gpu.alloc (%0, %1) : memref<?x?xf32>
    return %memref : memref<?x?xf32>
  }
  func.func @gpu_dealloc_2d_f32(%arg0: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    gpu.dealloc %arg0 : memref<?x?xf32>
    return
  }
  func.func @gpu_copy_2d_f32(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) attributes {llvm.emit_c_interface} {
    gpu.memcpy %arg1, %arg0 : memref<?x?xf32>, memref<?x?xf32>
    return
  }
  func.func @gpu_alloc_2d_f16(%arg0: i32, %arg1: i32) -> memref<?x?xf16> attributes {llvm.emit_c_interface} {
    %0 = arith.index_cast %arg0 : i32 to index
    %1 = arith.index_cast %arg1 : i32 to index
    %memref = gpu.alloc (%0, %1) : memref<?x?xf16>
    return %memref : memref<?x?xf16>
  }
  func.func @gpu_dealloc_2d_f16(%arg0: memref<?x?xf16>) attributes {llvm.emit_c_interface} {
    gpu.dealloc %arg0 : memref<?x?xf16>
    return
  }
  func.func @gpu_copy_2d_f16(%arg0: memref<?x?xf16>, %arg1: memref<?x?xf16>) attributes {llvm.emit_c_interface} {
    gpu.memcpy %arg1, %arg0 : memref<?x?xf16>, memref<?x?xf16>
    return
  }
}
