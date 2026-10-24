// RUN: inter-opt %s --inter-insert-sync | inter-translate --xemachine-to-ged - -o %t
// RUN: inter-ged-dump %t | FileCheck %s

func.func @k() attributes {xemachine.has_dpas = true, xemachine.target = #xemachine.target<chip = "bmg">} {
  %a = xemachine.archreg 0 : !xemachine.reg<64, 20>
  %b = xemachine.archreg 0 : !xemachine.reg<128, 24>
  %acc = xemachine.archreg 0 : !xemachine.reg<128, 32>
  // CHECK: pc=0 opcode=dpas exec=16 swsb=0xc0{{.*}}depth=8 repeat=8 bPrecision=f16 aPrecision=f16
  %result = xemachine.dpas %a, %b, %acc {aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32, repeatCount = 8 : i32, systolicDepth = 8 : i32} : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>, !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  // CHECK-NEXT: pc=16 opcode=dpas exec=16 swsb=0xc0
  %chained = xemachine.dpas %a, %b, %result {aPrecision = 0 : i32, bPrecision = 0 : i32, elemType = f32, repeatCount = 8 : i32, systolicDepth = 8 : i32} : (!xemachine.reg<64, 20>, !xemachine.reg<128, 24>, !xemachine.reg<128, 32>) -> !xemachine.reg<128, 32>
  // CHECK-NEXT: pc=32 opcode=sync exec=1 swsb=0x80 function=nop
  // CHECK-NEXT: pc=48 opcode=mov exec=32 swsb=0x0
  %copy = xemachine.mov %chained {execSize = 32 : i32} : (!xemachine.reg<128, 32>, i32) -> !xemachine.reg<32, 40>
  return
}
