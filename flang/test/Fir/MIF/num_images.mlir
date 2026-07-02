// RUN: fir-opt --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST"} {
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
    %2:2 = hlfir.declare %1 {uniq_name = "_QFEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %3 = fir.alloca i32 {bindc_name = "team_number", uniq_name = "_QFEteam_number"}
    %4:2 = hlfir.declare %3 {uniq_name = "_QFEteam_number"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    %5 = mif.num_images : () -> i32
    hlfir.assign %5 to %2#0 : i32, !fir.ref<i32>
    %6 = fir.load %4#0 : !fir.ref<i32>
    %7 = mif.num_images team_number %6 : (i32) -> i32
    hlfir.assign %7 to %2#0 : i32, !fir.ref<i32>
    return
  }


// CHECK-LABEL: func.func @_QQmain
// CHECK: fir.call @_QMprifPprif_num_images(
// CHECK: fir.call @_QMprifPprif_num_images_with_team_number(
