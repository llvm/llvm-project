// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

acc.private.recipe @privatization_ref_i64 : !fir.ref<i64> init {
^bb0(%arg0: !fir.ref<i64>):
  %0 = fir.alloca i64
  acc.yield %0 : !fir.ref<i64>
}

// CHECK-LABEL: func.func @private_i64
// CHECK: acc.loop control([[IV:%.+]] : i64)
// CHECK: [[ALLOC:%.+]] = fir.alloca i64
// CHECK: [[DECL:%.+]] = fir.declare [[ALLOC]] {uniq_name = "_private_arg0"}
// CHECK: fir.store [[IV]] to [[DECL]]

func.func @private_i64(%arg0 : !fir.ref<i64>) {
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %priv = acc.private varPtr(%arg0 : !fir.ref<i64>) recipe(@privatization_ref_i64) -> !fir.ref<i64> {implicit = true, name = ""}
  acc.loop private(%priv : !fir.ref<i64>) control(%siv : i64) = (%c1_i32 : i32) to (%c16_i32 : i32) step (%c1_i32 : i32) {
    %priv_decl = fir.declare %priv {uniq_name = "_private_arg0"} : (!fir.ref<i64>) -> !fir.ref<i64>
    fir.store %siv to %priv_decl : !fir.ref<i64>
    acc.yield
  } attributes {independent = [#acc.device_type<none>]}
  return
}

// CHECK-LABEL: func.func @par_private_i64
// CHECK: acc.parallel {
// CHECK: [[ALLOC:%.+]] = fir.alloca i64
// CHECK: [[DECL:%.+]] = fir.declare [[ALLOC]] {uniq_name = "_private_arg0"}
// CHECK: acc.loop control([[IV:%.+]] : i64)
// CHECK: fir.store [[IV]] to [[DECL]]

func.func @par_private_i64(%arg0 : !fir.ref<i64>) {
  %c16_i32 = arith.constant 16 : i32
  %c1_i32 = arith.constant 1 : i32
  %priv = acc.private varPtr(%arg0 : !fir.ref<i64>) recipe(@privatization_ref_i64) -> !fir.ref<i64> {implicit = true, name = ""}
  acc.parallel private(%priv : !fir.ref<i64>) {
    %priv_decl = fir.declare %priv {uniq_name = "_private_arg0"} : (!fir.ref<i64>) -> !fir.ref<i64>
    acc.loop control(%siv : i64) = (%c1_i32 : i32) to (%c16_i32 : i32) step (%c1_i32 : i32) {
      fir.store %siv to %priv_decl : !fir.ref<i64>
      acc.yield
    } attributes {independent = [#acc.device_type<none>]}
    acc.yield
  }
  return
}
