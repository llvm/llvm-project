// RUN: fir-opt --strict-fir-volatile-verifier %s --bufferize-hlfir --convert-hlfir-to-fir | FileCheck %s
func.func private @_QFPa() -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFFaEa"}
  %1 = fir.volatile_cast %0 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFaEa"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %c1_i32 = arith.constant 1 : i32
  hlfir.assign %c1_i32 to %2#0 : i32, !fir.ref<i32, volatile>
  %3 = fir.volatile_cast %2#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  return %4 : i32
}
func.func private @_QFPb() -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFFbEr"}
  %1 = fir.volatile_cast %0 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFbEr"} : (!fir.ref<i32, volatile>) -> (!fir.ref<i32, volatile>, !fir.ref<i32, volatile>)
  %c2_i32 = arith.constant 2 : i32
  hlfir.assign %c2_i32 to %2#0 : i32, !fir.ref<i32, volatile>
  %3 = fir.volatile_cast %2#0 : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
  %4 = fir.load %3 : !fir.ref<i32>
  return %4 : i32
}
func.func private @_QFPc() -> f32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
  %0 = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFFcEr"}
  %1 = fir.volatile_cast %0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
  %2:2 = hlfir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFcEr"} : (!fir.ref<f32, volatile>) -> (!fir.ref<f32, volatile>, !fir.ref<f32, volatile>)
  %cst = arith.constant 3.000000e+00 : f32
  hlfir.assign %cst to %2#0 : f32, !fir.ref<f32, volatile>
  %3 = fir.volatile_cast %2#0 : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
  %4 = fir.load %3 : !fir.ref<f32>
  return %4 : f32
}


// CHECK-LABEL:   func.func private @_QFPa() -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "a", uniq_name = "_QFFaEa"}
// CHECK:           %[[VAL_1:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFaEa"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_3:.*]] = arith.constant 1 : i32
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPb() -> i32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "r", uniq_name = "_QFFbEr"}
// CHECK:           %[[VAL_1:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFbEr"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
// CHECK:           %[[VAL_3:.*]] = arith.constant 2 : i32
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<i32, volatile>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<i32>
// CHECK:           return %[[VAL_5]] : i32
// CHECK:         }

// CHECK-LABEL:   func.func private @_QFPc() -> f32 attributes {fir.host_symbol = @_QQmain, llvm.linkage = #llvm.linkage<internal>} {
// CHECK:           %[[VAL_0:.*]] = fir.alloca f32 {bindc_name = "r", uniq_name = "_QFFcEr"}
// CHECK:           %[[VAL_1:.*]] = fir.volatile_cast %[[VAL_0]] : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
// CHECK:           %[[VAL_2:.*]] = fir.declare %[[VAL_1]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QFFcEr"} : (!fir.ref<f32, volatile>) -> !fir.ref<f32, volatile>
// CHECK:           %[[VAL_3:.*]] = arith.constant 3.000000e+00 : f32
// CHECK:           fir.store %[[VAL_3]] to %[[VAL_2]] : !fir.ref<f32, volatile>
// CHECK:           %[[VAL_4:.*]] = fir.volatile_cast %[[VAL_2]] : (!fir.ref<f32, volatile>) -> !fir.ref<f32>
// CHECK:           %[[VAL_5:.*]] = fir.load %[[VAL_4]] : !fir.ref<f32>
// CHECK:           return %[[VAL_5]] : f32
// CHECK:         }

