// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

func.func @k() attributes {xemachine.has_dpas = true} {
  %a = xemachine.archreg 0 : !xemachine.reg<64, 20>
  %b = xemachine.archreg 0 : !xemachine.reg<128, 24>
  %acc = xemachine.archreg 0 : !xemachine.reg<128, 32>
  // CHECK: pc=0 opcode=dpas exec=16{{.*}}depth=8 repeat=8 bPrecision=f16 aPrecision=f16
  %result = xemachine.dpas %a, %b, %acc {aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32, repeatCount = 8 : i32, systolicDepth = 8 : i32} : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>, !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  return
}
