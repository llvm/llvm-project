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
    %3 = acc.firstprivate varPtr(%2 : !fir.ref<i32>) recipe(@firstprivatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "t"}
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
// CHECK: acc.parallel
// CHECK: %[[ALLOCA:.*]] = fir.alloca i32 {{.*}}acc.var_name = #acc.var_name<"t">
// CHECK: %[[FIRSTPRIVLOAD:.*]] = fir.load %{{.*}} : !fir.ref<i32>
// CHECK: fir.store %[[FIRSTPRIVLOAD]] to %[[ALLOCA]] : !fir.ref<i32>
// CHECK: %[[ALLOCALOAD:.*]] = fir.load %[[ALLOCA]] : !fir.ref<i32>
// CHECK: %[[ADDI:.*]] = arith.addi %[[ALLOCALOAD]], %c1{{.*}} : i32
// CHECK: fir.store %[[ADDI]] to %[[ALLOCA]] : !fir.ref<i32>

// CHECK-MAP-LABEL: func.func @firstpriv
// CHECK-MAP: fir.load {{.*}} : !fir.ref<i32>
// CHECK-MAP: acc.parallel {
// CHECK-MAP-NOT: acc.firstprivate_map
// CHECK-MAP: fir.alloca i32 {{.*}}acc.var_name = #acc.var_name<"t">
// CHECK-MAP: fir.store {{.*}} to {{.*}} : !fir.ref<i32>
// CHECK-MAP: arith.addi {{.*}} %c1
