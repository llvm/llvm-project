// RUN: not inter-translate --xemachine-to-ged %s -o /dev/null 2>&1 | FileCheck %s

// CHECK: unsupported machine data type 'f16'
func.func @k() attributes {xemachine.target = #xemachine.target<chip = "bmg">} {
  %zero = xemachine.imm 0 : f16
  %result = xemachine.mov %zero : (!xemachine.imm, f16) -> !xemachine.reg<16, 4>
  return
}
