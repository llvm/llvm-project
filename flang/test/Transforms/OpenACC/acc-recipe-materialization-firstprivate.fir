// RUN: fir-opt %s -acc-recipe-materialization | FileCheck %s
// RUN: fir-opt %s -acc-recipe-materialization -acc-optimize-firstprivate-map | FileCheck %s --check-prefix=CHECK-MAP

module {
  acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
  ^bb0(%arg0: !fir.ref<i32>):
    %0 = fir.alloca i32
    acc.yield %0 : !fir.ref<i32>
  }
  acc.firstprivate.recipe @firstprivatization_ref_i32 : !fir.ref<i32> init {
  ^bb0(%arg0: !fir.ref<i32>):
    %0 = fir.alloca i32
    acc.yield %0 : !fir.ref<i32>
  } copy {
  ^bb0(%arg0: !fir.ref<i32>, %arg1: !fir.ref<i32>):
    %0 = fir.load %arg0 : !fir.ref<i32>
    fir.store %0 to %arg1 : !fir.ref<i32>
    acc.terminator
  }
  func.func @firstpriv() {
    %c1336_i32 = arith.constant 1336 : i32
    %0 = fir.dummy_scope : !fir.dscope
    %1 = fir.alloca i32 {bindc_name = "t", uniq_name = "_QFfirstprivEt"}
    %2 = fir.declare %1 {uniq_name = "_QFfirstprivEt"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.store %c1336_i32 to %2 : !fir.ref<i32>
    %3 = acc.firstprivate varPtr(%2 : !fir.ref<i32>) recipe(@firstprivatization_ref_i32) implicit(true) name("t") -> !fir.ref<i32>
    acc.parallel firstprivate(%3 : !fir.ref<i32>) {
      %c1_i32 = arith.constant 1 : i32
      %4 = fir.load %3 : !fir.ref<i32>
      %5 = arith.addi %4, %c1_i32 : i32
      fir.store %5 to %3 : !fir.ref<i32>
      acc.yield
    }
    return
  }
}

// Verify that the firstprivate was materialized into a copy outside the kernel
// and an alloca (as per the recipe) inside the region.
// Then ensure that all uses are of the private alloca.
// CHECK-LABEL: func.func @firstpriv
// CHECK:       [[DECL:%.*]] = fir.declare
// CHECK:       [[MAP:%.*]] = acc.firstprivate_map varPtr([[DECL]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "t"}
// CHECK:       acc.parallel {
// CHECK:       [[FIRSTPRIV:%.*]] = fir.alloca i32 {{.*}}acc.var_name = #acc.var_name<"t">
// CHECK:       [[FIRSTPRIVLOAD:%.*]] = fir.load [[MAP]] : !fir.ref<i32>
// CHECK:       fir.store [[FIRSTPRIVLOAD]] to [[FIRSTPRIV]] : !fir.ref<i32>
// CHECK:       [[FIRSTPRIVLOAD:%.*]] = fir.load [[FIRSTPRIV]] : !fir.ref<i32>
// CHECK:       [[ADDI:%.*]] = arith.addi [[FIRSTPRIVLOAD]], %c1{{.*}} : i32
// CHECK:       fir.store [[ADDI]] to [[FIRSTPRIV]] : !fir.ref<i32>
// CHECK-NEXT:  [[VALUELOAD:%.*]] = fir.load
// CHECK-NEXT:  acc.firstprivate_save [[VALUELOAD]] to [[MAP]] : i32 -> !fir.ref<i32>

// With -acc-optimize-firstprivate-map, the map is retained because of the
// firstprivate_save live-out.
// CHECK-MAP-LABEL: func.func @firstpriv
// CHECK-MAP:       [[DECL:%.*]] = fir.declare
// CHECK-MAP:       [[MAP:%.*]] = acc.firstprivate_map varPtr([[DECL]] : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "t"}
// CHECK-MAP:       acc.parallel {
// CHECK-MAP:       [[FIRSTPRIV:%.*]] = fir.alloca i32 {{.*}}acc.var_name = #acc.var_name<"t">
// CHECK-MAP:       fir.load [[MAP]] : !fir.ref<i32>
// CHECK-MAP:       fir.store {{.*}} to [[FIRSTPRIV]] : !fir.ref<i32>
// CHECK-MAP:       arith.addi {{.*}} %c1
// CHECK-MAP:       acc.firstprivate_save {{.*}} to [[MAP]] : i32 -> !fir.ref<i32>
