// RUN: inter-translate --xemachine-to-ged %s -o %t

module {
  func.func private @decoy()

  func.func @kernel() attributes {
      xemachine.target = #xemachine.target<chip = "bmg">
    } {
    %token = xemachine.token
    %r0 = xemachine.archreg 0 : !xemachine.reg<16, 0>
    xemachine.eot %r0 dep %token : !xemachine.reg<16, 0>
    return
  }
}
