// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefixes=ALL_MLIR,UNCHANGED_MLIR
// RUN: fir-opt --target-rewrite="target-features=+gfx9-insts,+wavefrontsize64" %s | FileCheck %s --check-prefixes=ALL_MLIR,CHANGED_MLIR

// RUN: tco %s | FileCheck %s --check-prefixes=ALL_LLVM,UNCHANGED_LLVM
// RUN: tco -target-features=+gfx9-insts,+wavefrontsize64 %s | FileCheck %s --check-prefixes=ALL_LLVM,CHANGED_LLVM


// Check MLIR output from the 'fir-opt' tool

// ALL_MLIR: module attributes {
// ALL_MLIR-SAME: fir.target_features = #llvm.target_features<[

// UNCHANGED_MLIR-SAME: "+sse"
// CHANGED_MLIR-SAME: "+gfx9-insts", "+wavefrontsize64"

// ALL_MLIR-SAME: ]>

// ALL_MLIR: func.func @dummyfunc() attributes {
// ALL_MLIR-SAME: target_features = #llvm.target_features<[

// UNCHANGED_MLIR-SAME: "+sse"
// CHANGED_MLIR-SAME: "+gfx9-insts", "+wavefrontsize64"

// ALL_MLIR-SAME: ]>


// Check LLVM output from the 'tco' tool

// ALL_LLVM: define void @dummyfunc() #[[ATTRS:.*]] {
// ALL_LLVM: attributes #[[ATTRS]] = {

// UNCHANGED_LLVM-SAME: "target-features"="+sse"
// CHANGED_LLVM-SAME: "target-features"="+gfx9-insts,+wavefrontsize64"

module attributes {fir.target_features = #llvm.target_features<["+sse"]>}  {
  func.func @dummyfunc() -> () {
    return
  }
}
