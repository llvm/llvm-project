// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

// CHECK: pc=0 opcode=sync exec=1 swsb=0x0 function=nop mask=normal channel=0 pred=normal
// CHECK: pc=16 opcode=sync exec=1 swsb=0x0 function=allwr mask=normal channel=0 pred=normal
func.func @k() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  %token = xemachine.sync nop : !xemachine.mem.token
  %wait = xemachine.sync allwr {sbidMask = 257 : i32} : !xemachine.mem.token
  return
}
