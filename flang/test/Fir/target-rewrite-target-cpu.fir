// RUN: fir-opt --target-rewrite %s | FileCheck %s --check-prefixes=ALL_AIIR,UNCHANGED_AIIR
// RUN: fir-opt --target-rewrite="target-cpu=gfx90a" %s | FileCheck %s --check-prefixes=ALL_AIIR,CHANGED_AIIR

// RUN: tco %s | FileCheck %s --check-prefixes=ALL_LLVM,UNCHANGED_LLVM
// RUN: tco -target-cpu=gfx90a %s | FileCheck %s --check-prefixes=ALL_LLVM,CHANGED_LLVM


// Check AIIR output from the 'fir-opt' tool

// ALL_AIIR: module attributes {
// ALL_AIIR-SAME: fir.target_cpu =

// UNCHANGED_AIIR-SAME: "x86_64"
// CHANGED_AIIR-SAME: "gfx90a"

// ALL_AIIR: func.func @dummyfunc() attributes {
// ALL_AIIR-SAME: target_cpu =

// UNCHANGED_AIIR-SAME: "x86_64"
// CHANGED_AIIR-SAME: "gfx90a"


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
