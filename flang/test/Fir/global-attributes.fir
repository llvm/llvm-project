// RUN: fir-opt --fir-to-llvm-ir %s | FileCheck %s
// RUN: tco --fir-to-llvm-ir %s | FileCheck %s
// RUN: fir-opt %s | FileCheck %s --check-prefix=READ-OUT
// RUN: tco --emit-fir %s | FileCheck %s --check-prefix=READ-OUT

// CHECK: llvm.mlir.global external @_QMtest_0Edata_int() {{{.*}}test = "string_attribute_maintained"{{.*}}} : i32 {
// CHECK: [[CST0:%.*]] = llvm.mlir.constant(10 : i32) : i32
// CHECK: llvm.return [[CST0]] : i32
// CHECK: }
// READ-OUT: fir.global @_QMtest_0Edata_int {test = "string_attribute_maintained"} : i32 {
// READ-OUT: %c10_i32 = arith.constant 10 : i32
// READ-OUT: fir.has_value %c10_i32 : i32
// READ-OUT: }
fir.global @_QMtest_0Edata_int {test = "string_attribute_maintained"} : i32 {
  %c10_i32 = arith.constant 10 : i32
  fir.has_value %c10_i32 : i32
}
