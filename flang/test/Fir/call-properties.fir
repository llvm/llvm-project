// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// Preserve the direct callee alongside converted fast-math properties and
// discardable attributes. An indirect call must keep its function operand.
func.func private @callee(f32) -> f32

func.func @direct(%arg: f32) -> f32 {
  %result = fir.call @callee(%arg) fastmath<contract> {test.tag = "direct"} : (f32) -> f32
  return %result : f32
}

// CHECK-LABEL: llvm.func @direct(
// CHECK: llvm.call @callee({{.*}}) {{.*}}fastmathFlags = #llvm.fastmath<contract>{{.*}}test.tag = "direct"

func.func @indirect(%callee: (f32) -> f32, %arg: f32) -> f32 {
  %result = fir.call %callee(%arg) fastmath<contract> {test.tag = "indirect"} : (f32) -> f32
  return %result : f32
}

// CHECK-LABEL: llvm.func @indirect(
// CHECK: llvm.call %{{.*}}({{.*}}) {{.*}}fastmathFlags = #llvm.fastmath<contract>{{.*}}test.tag = "indirect"
