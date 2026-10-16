// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s

module {
  func.func @private_load(%address: !xw.ptr<#xw.private>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "private", access = "read_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 8 : i32} {
    %root = xw.token : !xw.mem.token
    // expected-error@+1 {{unsupported XW load address space}}
    %value, %loaded = xw.load %address after %root : (!xw.ptr<#xw.private>, !xw.mem.token) -> (!xw.simd<i32, 8>, !xw.mem.token)
    return
  }
}

// -----

module {
  func.func @wide_store(%address: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "write_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 8 : i32} {
    %zero = xw.constant 0 : i64
    %value = xw.splat %zero : i64 -> !xw.simd<i64, 8>
    %root = xw.token : !xw.mem.token
    // expected-error@+1 {{only dword Xe memory stores are supported}}
    %stored = xw.store %value -> %address after %root : (!xw.simd<i64, 8>, !xw.ptr<#xw.global>, !xw.mem.token) -> !xw.mem.token
    return
  }
}

// -----

module {
  func.func @local_atomic() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %address = xw.local_memory_base : !xw.ptr<#xw.local>
    %one = xw.constant 1 : i32
    %value = xw.splat %one : i32 -> !xw.simd<i32, 8>
    %root = xw.token : !xw.mem.token
    // expected-error@+1 {{atomic add requires i32 data and an A64 address}}
    %old, %atomic = xw.atomic_rmw addi %value, %address after %root : (!xw.simd<i32, 8>, !xw.ptr<#xw.local>, !xw.mem.token) -> (!xw.simd<i32, 8>, !xw.mem.token)
    return
  }
}

// -----

module {
  func.func @local_i64_offset() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %base = xw.local_memory_base : !xw.ptr<#xw.local>
    %offset = xw.constant 4 : i64 -> !xw.simd<i64, 8>
    // expected-error@+1 {{local pointer offset must be i32}}
    %address = xw.ptradd %base, %offset : !xw.ptr<#xw.local>, !xw.simd<i64, 8> -> !xw.simd<!xw.ptr<#xw.local>, 8>
    return
  }
}
