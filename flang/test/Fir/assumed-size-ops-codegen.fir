// RUN: fir-opt --fir-to-llvm-ir="target=x86_64-unknown-linux-gnu" %s | FileCheck %s

// CHECK-LABEL: @assumed_size_extent(
// CHECK: %[[CNEG1:.*]] = llvm.mlir.constant(-1 : i64)
// CHECK: llvm.return %[[CNEG1]] : i64
func.func @assumed_size_extent() -> index {
  %e = fir.assumed_size_extent : index
  return %e : index
}

// CHECK-LABEL:  @is_assumed_size_extent(
// CHECK: %[[NEG1:.*]] = llvm.mlir.constant(-1 : i64)
// CHECK: %[[CMP:.*]] = llvm.icmp "eq"
// CHECK: llvm.return %[[CMP]] : i1
func.func @is_assumed_size_extent(%x: index) -> i1 {
  %c = fir.is_assumed_size_extent %x : (index) -> i1
  return %c : i1
}

