// Test lowering of REAL operations from FIR to LLVM IR

// RUN: tco %s | FileCheck %s

// CHECK-LABEL: @bar
func @bar(%a : !fir.real<2>, %b : !fir.real<4>, %c : !fir.real<8>, %d : !fir.real<10>, %e : !fir.real<16>) -> !fir.real<10> {
  // CHECK: fpext half %{{.*}} to x86_fp80
  %1 = fir.convert %a : (!fir.real<2>) -> !fir.real<10>
  // CHECK: fpext float %{{.*}} to x86_fp80
  %2 = fir.convert %b : (!fir.real<4>) -> !fir.real<10>
  // CHECK: fpext double %{{.*}} to x86_fp80
  %3 = fir.convert %c : (!fir.real<8>) -> !fir.real<10>
  // CHECK-NOT: fpext
  // CHECK-NOT: fptrunc
  %4 = fir.convert %d : (!fir.real<10>) -> !fir.real<10>
  // CHECK: fptrunc fp128 %{{.*}} to x86_fp80
  %5 = fir.convert %e : (!fir.real<16>) -> !fir.real<10>
  // CHECK-NEXT: call x86_fp80
  %6 = call @foop(%1, %2, %3, %4, %5) : (!fir.real<10>, !fir.real<10>, !fir.real<10>, !fir.real<10>, !fir.real<10>) -> !fir.real<10>
  return %6 : !fir.real<10>
}

// CHECK-LABEL: @foo
func @foo(%a : !fir.real<16>, %b : !fir.real<16>, %c : !fir.real<16>, %d : !fir.real<16>, %e : !fir.real<16>) -> !fir.real<16> {
  // CHECK: fadd fp128
  %1 = fir.addf %a, %b : !fir.real<16>
  // CHECK: fmul fp128
  %2 = fir.mulf %1, %c : !fir.real<16>
  // CHECK: fsub fp128
  %3 = fir.subf %2, %d : !fir.real<16>
  // CHECK: fdiv fp128
  %4 = fir.divf %3, %e : !fir.real<16>
  // CHECK: frem fp128
  %5 = fir.modf %4, %a : !fir.real<16>
  return %5 : !fir.real<16>
}

// CHECK-LABEL: @foop
func @foop(%a : !fir.real<10>, %b : !fir.real<10>, %c : !fir.real<10>, %d : !fir.real<10>, %e : !fir.real<10>) -> !fir.real<10> {
  // CHECK: fadd x86_fp80
  %1 = fir.addf %a, %b : !fir.real<10>
  // CHECK: fmul x86_fp80
  %2 = fir.mulf %1, %c : !fir.real<10>
  // CHECK: fsub x86_fp80
  %3 = fir.subf %2, %d : !fir.real<10>
  // CHECK: fdiv x86_fp80
  %4 = fir.divf %3, %e : !fir.real<10>
  // CHECK: frem x86_fp80
  %5 = fir.modf %4, %a : !fir.real<10>
  return %5 : !fir.real<10>
}
