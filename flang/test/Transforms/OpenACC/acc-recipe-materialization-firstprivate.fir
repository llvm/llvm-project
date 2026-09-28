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

  acc.firstprivate.recipe @firstprivatization_pointer : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> init {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
    %c4 = arith.constant 4 : index
    %alloc = fir.allocmem !fir.array<?xf32>, %c4
    %shape = fir.shape %c4 : (index) -> !fir.shape<1>
    %box = fir.embox %alloc(%shape) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.ptr<!fir.array<?xf32>>>
    %desc = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
    fir.store %box to %desc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    acc.yield %desc, %alloc : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.heap<!fir.array<?xf32>>
  } copy {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
       %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
    acc.terminator
  } destroy {
  ^bb0(%arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
       %arg1: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
       %owned: !fir.heap<!fir.array<?xf32>>):
    fir.freemem %owned : !fir.heap<!fir.array<?xf32>>
    acc.terminator
  }

  func.func @firstprivate_pointer(
      %arg0: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>,
      %arg1: !fir.box<!fir.ptr<!fir.array<?xf32>>>) {
    %priv = acc.firstprivate varPtr(%arg0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) recipe(@firstprivatization_pointer) name("p") -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
    acc.parallel firstprivate(%priv : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) {
      fir.store %arg1 to %priv : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
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

// CHECK-LABEL: func.func @firstprivate_pointer
// CHECK: %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>
// CHECK: %[[DESC:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
// CHECK: fir.store %arg1 to %[[DESC]]
// CHECK-NOT: fir.box_addr
// CHECK: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xf32>>

// CHECK-MAP-LABEL: func.func @firstprivate_pointer
// CHECK-MAP: %[[ALLOC:.*]] = fir.allocmem !fir.array<?xf32>
// CHECK-MAP: fir.freemem %[[ALLOC]] : !fir.heap<!fir.array<?xf32>>
