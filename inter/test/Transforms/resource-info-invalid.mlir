// RUN: inter-opt %s --split-input-file --inter-resource-info -verify-diagnostics

module {
  func.func @virtual_register() attributes {xemachine.grf_count = 128 : i32} {
    %zero = xemachine.imm 0 : i32
    // expected-error @below {{resource info requires physical XeMachine registers}}
    %virtual = xemachine.mov %zero : (!xemachine.imm, i32)
        -> !xemachine.reg<16, -1>
    return
  }
}

// -----

module {
  func.func @register_overflow() attributes {xemachine.grf_count = 128 : i32} {
    // expected-error @below {{physical register range ends at r129 but the selected GRF mode has 128 registers}}
    %overflow = xemachine.archreg 127 : !xemachine.reg<32, 127>
    return
  }
}

// -----

module {
  // expected-error @below {{resource info requires a positive xemachine.grf_count function attribute that fits in i32}}
  func.func @missing_grf_count() {
    return
  }
}

// -----

module {
  // expected-error @below {{resource info requires nonnegative xemachine.scratch_size when present}}
  func.func @negative_scratch() attributes {
      xemachine.grf_count = 128 : i32,
      xemachine.scratch_size = -1 : i64
    } {
    return
  }
}
