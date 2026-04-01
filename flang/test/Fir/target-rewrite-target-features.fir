// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefixes=ALL_AIIR,UNCHANGED_AIIR
// RUN: fir-opt --target-rewrite="target-features=+gfx9-insts,+wavefrontsize64" %s | FileCheck %s --check-prefixes=ALL_AIIR,CHANGED_AIIR

// RUN: tco %s | FileCheck %s --check-prefixes=ALL_LLVM,UNCHANGED_LLVM
// RUN: tco -target-features=+gfx9-insts,+wavefrontsize64 %s | FileCheck %s --check-prefixes=ALL_LLVM,CHANGED_LLVM


// Check AIIR output from the 'fir-opt' tool

// ALL_AIIR: module attributes {
// ALL_AIIR-SAME: fir.target_features = #llvm.target_features<[

// UNCHANGED_AIIR-SAME: "+sse"
// CHANGED_AIIR-SAME: "+gfx9-insts", "+wavefrontsize64"

// ALL_AIIR-SAME: ]>

// ALL_AIIR: func.func @dummyfunc() attributes {
// ALL_AIIR-SAME: target_features = #llvm.target_features<[

// UNCHANGED_AIIR-SAME: "+sse"
// CHANGED_AIIR-SAME: "+gfx9-insts", "+wavefrontsize64"

// ALL_AIIR-SAME: ]>


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
