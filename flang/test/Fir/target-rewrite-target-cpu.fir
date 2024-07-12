// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefixes=ALL_MLIR,UNCHANGED_MLIR
// RUN: fir-opt --target-rewrite="target-cpu=gfx90a" %s | FileCheck %s --check-prefixes=ALL_MLIR,CHANGED_MLIR

// RUN: tco %s | FileCheck %s --check-prefixes=ALL_LLVM,UNCHANGED_LLVM
// RUN: tco -target-cpu=gfx90a %s | FileCheck %s --check-prefixes=ALL_LLVM,CHANGED_LLVM


// Check MLIR output from the 'fir-opt' tool

// ALL_MLIR: module attributes {
// ALL_MLIR-SAME: fir.target_cpu =

// UNCHANGED_MLIR-SAME: "x86_64"
// CHANGED_MLIR-SAME: "gfx90a"

// ALL_MLIR: func.func @dummyfunc() attributes {
// ALL_MLIR-SAME: target_cpu =

// UNCHANGED_MLIR-SAME: "x86_64"
// CHANGED_MLIR-SAME: "gfx90a"


// Check LLVM output from the 'tco' tool

// ALL_LLVM: define void @dummyfunc() #[[ATTRS:.*]] {
// ALL_LLVM: attributes #[[ATTRS]] = {

// UNCHANGED_LLVM-SAME: "target-cpu"="x86_64"
// CHANGED_LLVM-SAME: "target-cpu"="gfx90a"

module attributes {fir.target_cpu = "x86_64"}  {
  func.func @dummyfunc() -> () {
    return
  }
}
