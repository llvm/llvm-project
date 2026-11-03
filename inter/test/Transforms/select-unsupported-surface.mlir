// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s
// CHECK-NOT: llvm

module {
  func.func @fmax() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 8>
    // expected-error@+1 {{floating maximum has no exact XeMachine primitive}}
    %result = xw.fmax %one, %one : !xw.simd<f32, 8>, !xw.simd<f32, 8> -> !xw.simd<f32, 8>
    return
  }
}

// -----

module {
  func.func @fma() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 8>
    // expected-error@+1 {{fused multiply-add has no exact XeMachine primitive}}
    %result = xw.fma %one, %one, %one : !xw.simd<f32, 8>, !xw.simd<f32, 8>, !xw.simd<f32, 8> -> !xw.simd<f32, 8>
    return
  }
}

// -----

module {
  func.func @fexp2() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 8>
    // expected-error@+1 {{base-two exponential has no exact XeMachine primitive}}
    %result = xw.fexp2 %one : !xw.simd<f32, 8> -> !xw.simd<f32, 8>
    return
  }
}

// -----

module {
  func.func @frcp() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1.0 : f32 -> !xw.simd<f32, 8>
    // expected-error@+1 {{reciprocal has no exact XeMachine primitive}}
    %result = xw.frcp %one : !xw.simd<f32, 8> -> !xw.simd<f32, 8>
    return
  }
}

// -----

module {
  func.func @lane_id() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    // expected-error@+1 {{lane ID has no XeMachine channel-index primitive}}
    %lane = xw.lane_id : !xw.simd<i32, 8>
    return
  }
}

// -----

module {
  func.func @global_size() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    // expected-error@+1 {{query is absent from the Xe payload contract}}
    %size = xw.global_size 0 : i32
    return
  }
}

// -----

module {
  func.func @num_groups() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    // expected-error@+1 {{query is absent from the Xe payload contract}}
    %groups = xw.num_groups 0 : i32
    return
  }
}

// -----

module {
  func.func @launch_grid_size() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    // expected-error@+1 {{query is absent from the Xe payload contract}}
    %grid = xw.launch_grid_size 0 : i32
    return
  }
}

// -----

module {
  func.func @dynamic_shuffle(%lane: i32) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 4, alignment = 4, offset = 24>],
      xw.simd_width = 8 : i32} {
    %one = xw.constant 1 : i32 -> !xw.simd<i32, 8>
    // expected-error@+1 {{dynamic shuffle has no XeMachine indirect-region primitive}}
    %result = xw.shuffle %one from %lane : !xw.simd<i32, 8>, i32 -> !xw.simd<i32, 8>
    return
  }
}

// -----

module {
  func.func @lossy_pointer(%global: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>],
      xw.simd_width = 8 : i32} {
    // expected-error@+1 {{A64 to local address-space cast would lose pointer bits}}
    %local = xw.addrspace_cast %global : !xw.ptr<#xw.global> -> !xw.ptr<#xw.local>
    return
  }
}

// -----

module {
  func.func @local_to_generic() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %local = xw.alloc() {bytesize = 32 : i64, align = 16 : i64}
        : !xw.ptr<#xw.local>
    // expected-error@+1 {{local and generic address-space casts lack provenance-preserving machine selection}}
    %generic = xw.addrspace_cast %local
        : !xw.ptr<#xw.local> -> !xw.ptr<#xw.generic>
    return
  }
}

// -----

module {
  func.func @unsupported_shift() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1 : i32 -> !xw.simd<i32, 8>
    // expected-error@+1 {{integer operation has no XeMachine instruction selection}}
    %result = xw.binary shrsi %one, %one : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.simd<i32, 8>
    return
  }
}

// -----

module {
  func.func @wide_division() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 32 : i32} {
    %one = xw.constant 1 : i64 -> !xw.simd<i64, 32>
    // expected-error@+1 {{SIMD32 i64 division/remainder has no exact two-half flag selection}}
    %result = xw.binary divui %one, %one : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.simd<i64, 32>
    return
  }
}

// -----

module {
  func.func @unsupported_mulhi() attributes {xemachine.kernel, xemachine.kernel_args = [], xw.simd_width = 8 : i32} {
    %one = xw.constant 1 : i32 -> !xw.simd<i32, 8>
    // expected-error@+1 {{integer operation has no XeMachine instruction selection}}
    %result = xw.binary mulhui %one, %one : !xw.simd<i32, 8>, !xw.simd<i32, 8> -> !xw.simd<i32, 8>
    return
  }
}

// -----

module {
  func.func @wide_compare() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 32 : i32} {
    %one = xw.constant 1 : i64 -> !xw.simd<i64, 32>
    // expected-error@+1 {{SIMD32 i64 comparison has no exact two-half flag selection}}
    %equal = xw.cmpi eq %one, %one
        : !xw.simd<i64, 32>, !xw.simd<i64, 32> -> !xw.mask<32>
    return
  }
}

// -----

module {
  func.func @wide_select() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 32 : i32} {
    %narrow = xw.constant 1 : i32 -> !xw.simd<i32, 32>
    %one = xw.constant 1 : i64 -> !xw.simd<i64, 32>
    %equal = xw.cmpi eq %narrow, %narrow
        : !xw.simd<i32, 32>, !xw.simd<i32, 32> -> !xw.mask<32>
    // expected-error@+1 {{SIMD32 i64 or A64 pointer select has no exact two-half selection}}
    %selected = xw.select %equal, %one, %one
        : !xw.mask<32>, !xw.simd<i64, 32>
    return
  }
}

// -----

module {
  func.func @unsupported_block2d(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %size = xw.constant 128 : i32
    %zero = xw.constant 0 : i32
    %root = xw.token : !xw.mem.token
    // expected-error@+1 {{BMG selection supports only an untransformed 16-bit 8x16 single-block prefetch}}
    %prefetched = xw.block2d_prefetch %base[%zero, %zero]
        surface (%size, %size, %size) after %root
        {block_height = 16 : i64, block_width = 16 : i64, blocks = 1 : i64,
         element_bits = 16 : i64}
        : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32, !xw.mem.token)
          -> !xw.mem.token
    return
  }
}

// -----

module {
  func.func @unsupported_block2d_read_packet(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %size = xw.constant 128 : i32
    %zero = xw.constant 0 : i32
    // expected-error@+1 {{result packet does not match the selected block2D read}}
    %value, %token = xw.block2d_read %base[%zero, %zero]
        surface (%size, %size, %size)
        {block_height = 8 : i64, block_width = 16 : i64, blocks = 1 : i64,
         element_bits = 16 : i64}
        : (!xw.ptr<#xw.global>, i32, i32, i32, i32, i32)
          -> (!xw.simd<vector<4xi16>, 16>, !xw.mem.token)
    return
  }
}

// -----

module {
  func.func @unsupported_block2d_write_packet(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 16 : i32} {
    %size = xw.constant 128 : i32
    %zero = xw.constant 0 : i32
    %value = xw.constant dense<0> : vector<4xi32>
        -> !xw.simd<vector<4xi32>, 16>
    // expected-error@+1 {{data packet does not match the selected block2D write}}
    %token = xw.block2d_write %value -> %base[%zero, %zero]
        surface (%size, %size, %size)
        {block_height = 8 : i64, block_width = 16 : i64, blocks = 1 : i64,
         element_bits = 32 : i64}
        : (!xw.simd<vector<4xi32>, 16>, !xw.ptr<#xw.global>, i32, i32,
           i32, i32, i32) -> !xw.mem.token
    return
  }
}
