// RUN: inter-opt %s --loop-invariant-code-motion | FileCheck %s

module {
  func.func @nested_uniform_licm(
      %condition: !xemachine.arf<f, 2, -1>,
      %base: !xemachine.reg<16, -1>,
      %dynamic: !xemachine.reg<1, -1>) {
    %loop = xemachine.uniform_loop(%dynamic) {
    ^bb0(%iter: !xemachine.reg<1, -1>):
      xemachine.uniform_if %condition : !xemachine.arf<f, 2, -1> {
        %constant = xemachine.imm 7 : i32
        %invariant = xemachine.mov %constant {execSize = 1 : i32, noMask}
            : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
        %template = xemachine.update_tuple %base, %invariant {offsets = [2]}
            : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>)
            -> !xemachine.reg<16, -1>
        %updated = xemachine.update_tuple %template, %iter {offsets = [5]}
            : (!xemachine.reg<16, -1>, !xemachine.reg<1, -1>)
            -> !xemachine.reg<16, -1>
        %masked = xemachine.mov %constant {execSize = 1 : i32}
            : (!xemachine.imm, i32) -> !xemachine.reg<1, -1>
        xemachine.yield
      }
      xemachine.continue_if %condition : !xemachine.arf<f, 2, -1>
          (%iter : !xemachine.reg<1, -1>)
    } : (!xemachine.reg<1, -1>) -> !xemachine.reg<1, -1>
    return
  }
}

// CHECK-LABEL: func.func @nested_uniform_licm
// CHECK: [[CONSTANT:%.*]] = xemachine.imm 7
// CHECK-NEXT: [[INVARIANT:%.*]] = xemachine.mov [[CONSTANT]] {{.*}}noMask
// CHECK-NEXT: [[TEMPLATE:%.*]] = xemachine.update_tuple {{%.*}}, [[INVARIANT]]
// CHECK-NEXT: xemachine.uniform_loop
// CHECK: xemachine.uniform_if
// CHECK-NOT: xemachine.imm 7
// CHECK: xemachine.update_tuple [[TEMPLATE]], {{%.*}}
// CHECK: xemachine.mov [[CONSTANT]]
// CHECK-NOT: noMask
