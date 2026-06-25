// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = mif.this_image : () -> i32
    hlfir.assign %3 to %2#0 : i32, !fir.ref<i32>
    return
  }


// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_this_image_no_coarray(
