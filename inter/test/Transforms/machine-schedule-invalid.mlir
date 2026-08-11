// RUN: inter-opt --split-input-file --inter-machine-schedule -verify-diagnostics %s

module {
  // expected-error@+1 {{machine scheduling requires a target attribute}}
  func.func @missing_target() {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    return
  }
}


// -----

module {
  func.func @unsupported_nested_region(%condition: i1) attributes {
      xemachine.target = #xemachine.target<chip = "bmg">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    // expected-error@+1 {{machine scheduler does not support nested region operation}}
    scf.if %condition {
    }
    return
  }
}

// -----

module {
  // expected-error@+1 {{machine scheduling does not support target 'xe3'}}
  func.func @unsupported_target() attributes {
      xemachine.target = #xemachine.target<chip = "xe3">} {
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    return
  }
}
