// Test lowering FIR to LLVM IR for a recursive type

// RUN: tco %s | FileCheck %s

// CHECK-LABEL: %t = type { %t* }
!t = type !fir.type<t {p : !fir.ptr<!fir.type<t>>}>

// CHECK-LABEL: @a(%t %{{.*}})
func @a(%a : !t) {
  return
}
