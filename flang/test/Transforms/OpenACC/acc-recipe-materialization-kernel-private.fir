// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s

// acc.kernels with private: recipe materialized to alloca inside region
// CHECK-NOT: acc.private
// CHECK: fir.alloca i32 {{.*}}acc.var_name = #acc.var_name<"s">

acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
^bb0(%arg0: !fir.ref<i32>):
  %0 = fir.alloca i32
  acc.yield %0 : !fir.ref<i32>
}
func.func @kpriv_(%arg0: !fir.ref<i32> {fir.bindc_name = "start"}, %arg1: !fir.ref<!fir.array<32xi32>> {fir.bindc_name = "a"}) attributes {fir.internal_name = "_QPkpriv"} {
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c32 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg1(%1) dummy_scope %0 arg 2 {uniq_name = "_QFkprivEa"} : (!fir.ref<!fir.array<32xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<32xi32>>
  %3 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFkprivEi"}
  %4 = fir.declare %3 {uniq_name = "_QFkprivEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %5 = fir.alloca i32 {bindc_name = "s", uniq_name = "_QFkprivEs"}
  %6 = fir.declare %5 {uniq_name = "_QFkprivEs"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %7 = fir.declare %arg0 dummy_scope %0 arg 1 {fortran_attrs = #fir.var_attrs<intent_in>, uniq_name = "_QFkprivEstart"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  %8 = fir.load %7 : !fir.ref<i32>
  fir.store %8 to %6 : !fir.ref<i32>
  %9 = acc.copyin varPtr(%2 : !fir.ref<!fir.array<32xi32>>) -> !fir.ref<!fir.array<32xi32>> {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "a"}
  %10 = acc.private varPtr(%6 : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "s"}
  acc.kernels dataOperands(%9 : !fir.ref<!fir.array<32xi32>>) private(%10 : !fir.ref<i32>) {
    %11 = fir.shape %c32 : (index) -> !fir.shape<1>
    acc.loop control(%arg2 : index) = (%c1 : index) to (%c32 : index) step (%c1 : index) {
      %12 = fir.alloca i32 {bindc_name = "i"}
      %13 = fir.declare %12 {uniq_name = "_QFkprivEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
      %14 = fir.convert %arg2 : (index) -> i32
      fir.store %14 to %13 : !fir.ref<i32>
      %15 = fir.load %10 : !fir.ref<i32>
      %16 = fir.load %13 : !fir.ref<i32>
      %17 = fir.convert %16 : (i32) -> i64
      %18 = fir.array_coor %9(%11) %17 : (!fir.ref<!fir.array<32xi32>>, !fir.shape<1>, i64) -> !fir.ref<i32>
      fir.store %15 to %18 : !fir.ref<i32>
      acc.yield
    } attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}
    acc.terminator
  }
  acc.copyout accPtr(%9 : !fir.ref<!fir.array<32xi32>>) to varPtr(%2 : !fir.ref<!fir.array<32xi32>>) {dataClause = #acc<data_clause acc_copy>, implicit = true, name = "a"}
  return
}
