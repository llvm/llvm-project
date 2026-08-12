// RUN: inter-opt %s | FileCheck %s

func.func @subgrf_update(%base: !xemachine.reg<32, -1>,
                         %update: !xemachine.reg<16, -1>) {
  %result = xemachine.update_tuple %base, %update {offsets = [8]}
      : (!xemachine.reg<32, -1>, !xemachine.reg<16, -1>)
      -> !xemachine.reg<32, -1>
  return
}

// CHECK: xemachine.update_tuple {{.*}} {offsets = [8]}
