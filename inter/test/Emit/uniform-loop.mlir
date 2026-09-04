// RUN: inter-translate --xemachine-to-ged %s -o %t
// RUN: inter-ged-dump %t | FileCheck %s
// RUN: inter-translate --xemachine-to-asm %s | FileCheck %s --check-prefix=ASM

func.func @loops() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  %one = xemachine.imm 1 : i32
  %init = xemachine.mov %one {execSize = 1 : i32, noMask}
      : (!xemachine.imm, i32) -> !xemachine.reg<16, 4>
  %outer = xemachine.arfreg f, 0 : !xemachine.arf<f, 2, 0>
  %inner = xemachine.arfreg f, 1 : !xemachine.arf<f, 2, 1>
  %result = xemachine.uniform_loop (%init) {
  ^bb0(%iter: !xemachine.reg<16, 4>):
    xemachine.uniform_loop () {
      %next = xemachine.add %iter, %one {execSize = 1 : i32, noMask}
          : (!xemachine.reg<16, 4>, !xemachine.imm, i32)
          -> !xemachine.reg<16, 4>
      xemachine.continue_if %inner : !xemachine.arf<f, 2, 1>
    } : () -> ()
    xemachine.continue_if %outer : !xemachine.arf<f, 2, 0>
        (%iter : !xemachine.reg<16, 4>)
  } : (!xemachine.reg<16, 4>) -> !xemachine.reg<16, 4>
  %use = xemachine.add %result, %one {execSize = 1 : i32, noMask}
      : (!xemachine.reg<16, 4>, !xemachine.imm, i32)
      -> !xemachine.reg<16, 5>
  return
}

// CHECK: pc=32 opcode=jmpi {{.*}}jip=-16 {{.*}}flag=1.0
// CHECK-NEXT: pc=48 opcode=jmpi {{.*}}jip=-32 {{.*}}flag=0.0
// CHECK-NEXT: pc=64 opcode=add
// ASM: L0:
// ASM: [[HEADER:L[0-9]+]]:
// ASM: (W&f1.0) jmpi [[HEADER]]
// ASM: (W&f0.0) jmpi [[HEADER]]
