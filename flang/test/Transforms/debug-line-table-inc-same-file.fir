
// RUN: fir-opt --add-debug-foundation --mlir-print-debuginfo %s | FileCheck %s
// REQUIRES: linux

// Test that there is only one FileAttribute generated for multiple functions
// in the same file.
module attributes {} {
  func.func @_QPs1() {
    return loc(#loc2)
  } loc(#loc1)
  func.func @_QPs2() {
    return loc(#loc2)
  } loc(#loc1)
  func.func @_QQmain() {
    return loc(#loc3)
  } loc(#loc1)
} loc(#loc)
#loc = loc("/home/user01/llvm-project/build_release/simple.f90":0:0)
#loc1 = loc("/home/user01/llvm-project/build_release/simple.f90":1:1)
#loc2 = loc("/home/user01/llvm-project/build_release/simple.f90":2:1)
#loc3 = loc("/home/user01/llvm-project/build_release/simple.f90":3:1)

// CHECK-COUNT-1: #llvm.di_file
