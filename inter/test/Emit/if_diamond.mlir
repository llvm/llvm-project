// exec_if lowers to native goto and join instructions with byte offsets.
// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

// CHECK: pc=0 opcode=goto exec=32 swsb=0x0 jip=32 uip=32 mask=normal channel=0 pred=sequential inverse=1 flag=0.0
// CHECK-NEXT: pc=16 opcode=goto exec=32 swsb=0x0 jip=16 uip=32 mask=normal channel=0 pred=normal
// CHECK-NEXT: pc=32 opcode=join exec=32 swsb=0x0 jip=16 mask=normal channel=0 pred=normal
// CHECK-NEXT: pc=48 opcode=join exec=32 swsb=0x0 jip=16 mask=normal channel=0 pred=normal
func.func @k(%f: !xemachine.arf<f, 2, 0>) {
  xemachine.exec_if %f : !xemachine.arf<f, 2, 0> {
    xemachine.yield
  } otherwise {
    xemachine.yield
  }
  return
}
