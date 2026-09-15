// RUN: inter-opt %s --split-input-file --verify-diagnostics

func.func @invalid_packet_widths() {
  %a = xemachine.archreg 4 : !xemachine.reg<16, 4>
  %b = xemachine.archreg 5 : !xemachine.reg<16, 5>
  %acc = xemachine.archreg 6 : !xemachine.reg<16, 6>
  // expected-error@+1 {{packet widths must match Xe2 depth and repeat count}}
  %result = xemachine.dpas %a, %b, %acc {
      aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32,
      repeatCount = 8 : i32, systolicDepth = 8 : i32}
      : (!xemachine.reg<16, 4>, !xemachine.reg<16, 5>,
         !xemachine.reg<16, 6>) -> !xemachine.reg<16, 6>
  return
}

// -----

func.func @invalid_depth() {
  %a = xemachine.archreg 4 : !xemachine.reg<32, 4>
  %b = xemachine.archreg 6 : !xemachine.reg<64, 6>
  %acc = xemachine.archreg 10 : !xemachine.reg<128, 10>
  // expected-error@+1 {{requires systolic depth 8 on Xe2}}
  %result = xemachine.dpas %a, %b, %acc {
      aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32,
      repeatCount = 8 : i32, systolicDepth = 4 : i32}
      : (!xemachine.reg<32, 4>, !xemachine.reg<64, 6>,
         !xemachine.reg<128, 10>) -> !xemachine.reg<128, 10>
  return
}
