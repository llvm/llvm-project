// RUN: inter-opt --split-input-file --inter-machine-schedule -verify-diagnostics %s

module {
  // expected-error@+1 {{missing Intel GPU target attribute}}
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
