// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

module {
  acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
  ^bb0(%arg0: !fir.ref<i32>):
    %0 = fir.alloca i32
    acc.yield %0 : !fir.ref<i32>
  }
  acc.firstprivate.recipe @firstprivatization_ref_rec__QFtestTpoint : !fir.ref<!fir.type<_QFtestTpoint{x:f32}>> init {
  ^bb0(%arg0: !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>):
    %0 = fir.alloca !fir.type<_QFtestTpoint{x:f32}>
    %1 = fir.declare %0 {uniq_name = "acc.private.init"} : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>
    acc.yield %1 : !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>
  } copy {
  ^bb0(%arg0: !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>, %arg1: !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>):
    %0 = fir.field_index x, !fir.type<_QFtestTpoint{x:f32}>
    %1 = fir.coordinate_of %arg0, x : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<f32>
    %2 = fir.field_index x, !fir.type<_QFtestTpoint{x:f32}>
    %3 = fir.coordinate_of %arg1, x : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<f32>
    %4 = fir.load %1 : !fir.ref<f32>
    fir.store %4 to %3 : !fir.ref<f32>
    acc.terminator
  }
  func.func @_QPtest(%arg0: !fir.ref<!fir.type<_QFtestTpoint{x:f32}>> {fir.bindc_name = "p"}) {
    %c1_i32 = arith.constant 1 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtestEi"}
    %2 = fir.declare %1 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.alloca i32 {bindc_name = "n", uniq_name = "_QFtestEn"}
    %4 = fir.declare %3 {uniq_name = "_QFtestEn"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %5 = fir.declare %arg0 dummy_scope %0 {uniq_name = "_QFtestEp"} : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>, !fir.dscope) -> !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>
    %6 = acc.firstprivate varPtr(%5 : !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) recipe(@firstprivatization_ref_rec__QFtestTpoint) -> !fir.ref<!fir.type<_QFtestTpoint{x:f32}>> {name = "p"}
    acc.parallel firstprivate(%6 : !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) {
      %7 = fir.load %4 : !fir.ref<i32>
      %8 = acc.private varPtr(%2 : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
      acc.loop private(%8 : !fir.ref<i32>) control(%arg1 : i32) = (%c1_i32 : i32) to (%7 : i32)  step (%c1_i32 : i32) {
        %9 = fir.declare %8 {uniq_name = "_QFtestEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
        fir.store %arg1 to %9 : !fir.ref<i32>
        %10 = fir.load %9 : !fir.ref<i32>
        %11 = fir.convert %10 : (i32) -> f32
        %12 = fir.field_index x, !fir.type<_QFtestTpoint{x:f32}>
        %13 = fir.coordinate_of %5, x : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<f32>
        fir.store %11 to %13 : !fir.ref<f32>
        acc.yield
      } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
      acc.yield
    }
    return
  }
}

// CHECK:           %[[VAL_7:.*]] = acc.firstprivate_map varPtr(%{{.*}} : !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>
// CHECK:           acc.parallel {
// CHECK:             %[[VAL_8:.*]] = fir.alloca !fir.type<_QFtestTpoint{x:f32}>
// CHECK:             %[[VAL_9:.*]] = fir.declare %[[VAL_8]] {acc.var_name = #acc.var_name<"p">, uniq_name = "acc.private.init"} : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<!fir.type<_QFtestTpoint{x:f32}>>
// CHECK:             %[[VAL_10:.*]] = fir.field_index x, !fir.type<_QFtestTpoint{x:f32}>
// CHECK:             %[[VAL_11:.*]] = fir.coordinate_of %[[VAL_7]], x : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<f32>
// CHECK:             %[[VAL_12:.*]] = fir.field_index x, !fir.type<_QFtestTpoint{x:f32}>
// CHECK:             %[[VAL_13:.*]] = fir.coordinate_of %[[VAL_9]], x : (!fir.ref<!fir.type<_QFtestTpoint{x:f32}>>) -> !fir.ref<f32>
// CHECK:             %[[VAL_14:.*]] = fir.load %[[VAL_11]] : !fir.ref<f32>
