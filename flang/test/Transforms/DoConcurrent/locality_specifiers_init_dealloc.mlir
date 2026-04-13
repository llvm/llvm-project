// Tests mapping `local` locality specifier to `private` clauses for non-empty
// `init` and `dealloc` regions.

// RUN: fir-opt --omp-do-concurrent-conversion="map-to=host" %s | FileCheck %s

func.func @my_allocator(%arg0: !fir.ref<!fir.box<!fir.array<10xf32>>>, %arg1: !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  return
}

func.func @my_deallocator(%arg0: !fir.ref<!fir.box<!fir.array<10xf32>>>) {
  return
}

fir.local {type = local} @_QFlocal_assocEaa_private_box_10xf32 : !fir.box<!fir.array<10xf32>> init {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<10xf32>>>, %arg1: !fir.ref<!fir.box<!fir.array<10xf32>>>):
  fir.call @my_allocator(%arg0, %arg1) : (!fir.ref<!fir.box<!fir.array<10xf32>>>, !fir.ref<!fir.box<!fir.array<10xf32>>>) -> ()
  fir.yield(%arg1 : !fir.ref<!fir.box<!fir.array<10xf32>>>)
} dealloc {
^bb0(%arg0: !fir.ref<!fir.box<!fir.array<10xf32>>>):
  fir.call @my_deallocator(%arg0) : (!fir.ref<!fir.box<!fir.array<10xf32>>>) -> ()
  fir.yield
}

func.func @_QPlocal_assoc() {
  %0 = fir.alloca !fir.box<!fir.array<10xf32>>
  %c1 = arith.constant 1 : index

  fir.do_concurrent {
    %9 = fir.alloca i32 {bindc_name = "i"}
    %10:2 = hlfir.declare %9 {uniq_name = "_QFlocal_assocEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
    fir.do_concurrent.loop (%arg0) = (%c1) to (%c1) step (%c1) local(@_QFlocal_assocEaa_private_box_10xf32 %0 -> %arg1 : !fir.ref<!fir.box<!fir.array<10xf32>>>) {
      %11 = fir.convert %arg0 : (index) -> i32
      fir.store %11 to %10#0 : !fir.ref<i32>
    }
  }

  return
}

// CHECK:      omp.private {type = private} @[[PRIVATIZER:.*]] : !fir.box<!fir.array<10xf32>> init {
// CHECK-NEXT: ^bb0(%[[ORIG_ARG:.*]]: !{{.*}}, %[[PRIV_ARG:.*]]: !{{.*}}):
// CHECK-NEXT:   fir.call @my_allocator(%[[ORIG_ARG]], %[[PRIV_ARG]]) : ({{.*}}) -> ()
// CHECK-NEXT:   omp.yield(%[[PRIV_ARG]] : {{.*}})
// CHECK-NEXT: } dealloc {
// CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !{{.*}}):
// CHECK-NEXT:   fir.call @my_deallocator(%[[PRIV_ARG]]) : ({{.*}}) -> ()
// CHECK-NEXT:   omp.yield
// CHECK-NEXT: }

// CHECK: %[[LOCAL_ALLOC:.*]] = fir.alloca !fir.box<!fir.array<10xf32>>
// CHECK: omp.wsloop private(@[[PRIVATIZER]] %[[LOCAL_ALLOC]] -> %{{.*}} : !{{.*}})
