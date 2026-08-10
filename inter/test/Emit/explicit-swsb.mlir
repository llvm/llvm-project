// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

func.func @k() {
  %root = xemachine.token
  %address = xemachine.archreg 0 : !xemachine.reg<16, 0>
  // CHECK: pc=0 opcode=send exec=1 swsb=0x5
  %dst, %token = xemachine.send ugm %address dep %root {desc = 136316288 : i32, exdesc = 0 : i32, noMask, sfid = 0 : i32, swsb = 5 : i64} : (!xemachine.reg<16, 0>) -> (!xemachine.reg<16, 4>, !xemachine.mem.token)
  return
}
