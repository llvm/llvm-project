// RUN: not inter-opt %s --inter-convert-llvm-to-xw --verify-each=false \
// RUN:   2>&1 | FileCheck %s

module {
  func.func @bad() {
    %value = "ub.poison"() <{value = unit}> : () -> i32
    return
  }
}

// CHECK: Invalid attribute `value` in property conversion: unit
