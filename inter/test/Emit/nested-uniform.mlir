// A uniform branch nested under a divergent mask uses mask-aware goto/join.
// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s

func.func @nested(%outer: !xemachine.arf<f, 2, 0>,
                   %inner: !xemachine.arf<f, 2, 1>) attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  xemachine.exec_if %outer : !xemachine.arf<f, 2, 0> {
    xemachine.uniform_if %inner : !xemachine.arf<f, 2, 1> {
      xemachine.yield
    } otherwise {
      xemachine.yield
    }
    xemachine.yield
  } otherwise {
    xemachine.yield
  }
  return
}

// CHECK: pc=0 opcode=goto {{.*}}flag=0.0
// CHECK-NEXT: pc=16 opcode=goto {{.*}}flag=1.0
// CHECK-NOT: opcode=jmpi
