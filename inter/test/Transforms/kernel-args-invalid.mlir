// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s

module {
  // expected-error@+1 {{by-value argument has pointer ABI properties}}
  func.func @wrong_kind(%arg: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_value, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>
       ], xw.simd_width = 8 : i32} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument crosses a payload boundary}}
  func.func @crosses_chunk(%arg: vector<4xi32>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 16, alignment = 8, offset = 88>
       ], xw.simd_width = 8 : i32} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument overlaps the implicit payload}}
  func.func @reserved(%arg: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 16>
       ], xw.simd_width = 8 : i32} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument payload is misaligned}}
  func.func @misaligned(%arg: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 28>
       ], xw.simd_width = 8 : i32} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument is outside the loaded payload}}
  func.func @out_of_bounds(%arg: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 192>
       ], xw.simd_width = 8 : i32} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument payloads overlap}}
  func.func @overlap(%pointer: !xw.ptr<#xw.global>, %value: i32) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_write", size = 8, alignment = 8, offset = 24>,
        #xemachine.kernel_arg<kind = by_value, address_space = "none", access = "none", size = 4, alignment = 4, offset = 28>
       ], xw.simd_width = 8 : i32} {
    return
  }
}
