// Test lowering of REAL operations from FIR to LLVM IR

// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @bar
func.func @bar(%a : f16, %b : f32, %c : f64, %d : f80, %e : f128) -> f80 {
  // CHECK: fpext half %{{.*}} to x86_fp80
  %1 = fir.convert %a : (f16) -> f80
  // CHECK: fpext float %{{.*}} to x86_fp80
  %2 = fir.convert %b : (f32) -> f80
  // CHECK: fpext double %{{.*}} to x86_fp80
  %3 = fir.convert %c : (f64) -> f80
  // CHECK-NOT: fpext
  // CHECK-NOT: fptrunc
  %4 = fir.convert %d : (f80) -> f80
  // CHECK: fptrunc fp128 %{{.*}} to x86_fp80
  %5 = fir.convert %e : (f128) -> f80
  // CHECK-NEXT: call x86_fp80
  %6 = fir.call @foop(%1, %2, %3, %4, %5) : (f80, f80, f80, f80, f80) -> f80
  return %6 : f80
}

// CHECK-LABEL: @foo
func.func @foo(%a : f128, %b : f128, %c : f128, %d : f128, %e : f128) -> f128 {
  // CHECK: fadd fp128
  %1 = arith.addf %a, %b : f128
  // CHECK: fmul fp128
  %2 = arith.mulf %1, %c : f128
  // CHECK: fsub fp128
  %3 = arith.subf %2, %d : f128
  // CHECK: fdiv fp128
  %4 = arith.divf %3, %e : f128
  // CHECK: frem fp128
  %5 = arith.remf %4, %a : f128
  return %5 : f128
}

// CHECK-LABEL: @foop
func.func @foop(%a : f80, %b : f80, %c : f80, %d : f80, %e : f80) -> f80 {
  // CHECK: fadd x86_fp80
  %1 = arith.addf %a, %b : f80
  // CHECK: fmul x86_fp80
  %2 = arith.mulf %1, %c : f80
  // CHECK: fsub x86_fp80
  %3 = arith.subf %2, %d : f80
  // CHECK: fdiv x86_fp80
  %4 = arith.divf %3, %e : f80
  // CHECK: frem x86_fp80
  %5 = arith.remf %4, %a : f80
  return %5 : f80
}
