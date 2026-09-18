// exec_if lowers to native goto and join instructions with byte offsets.
// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s
// RUN: inter-translate --xemachine-to-asm %s | FileCheck %s --check-prefix=ASM

// CHECK: pc=0 opcode=goto exec=32 swsb=0x0 jip=32 uip=32 mask=normal channel=0 pred=sequential inverse=1 flag=0.0
// CHECK-NEXT: pc=16 opcode=goto exec=32 swsb=0x0 jip=16 uip=32 mask=normal channel=0 pred=normal
// CHECK-NEXT: pc=32 opcode=join exec=32 swsb=0x0 jip=16 mask=normal channel=0 pred=normal
// CHECK-NEXT: pc=48 opcode=join exec=32 swsb=0x0 jip=16 mask=normal channel=0 pred=normal
func.func @k(%f: !xemachine.arf<f, 2, 0>) attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  xemachine.exec_if %f : !xemachine.arf<f, 2, 0> {
    xemachine.yield
  } otherwise {
    xemachine.yield
  }
  return
}

// ASM: [[ENTRY:L[0-9]+]]:
// ASM-NEXT: (~f0.0) goto (32|M0) [[THEN:L[0-9]+]] [[THEN]]
// ASM-NEXT: [[ELSE:L[0-9]+]]:
// ASM-NEXT: goto (32|M0) [[THEN]] [[EXIT:L[0-9]+]]
// ASM-NEXT: [[THEN]]:
// ASM-NEXT: join (32|M0) [[EXIT]]
// ASM-NEXT: [[EXIT]]:
// ASM-NEXT: join (32|M0) [[END:L[0-9]+]]
