// Test overlapping assignment of derived type arrays with allocatable components.
// This requires initializing the allocatable components to an unallocated status
// before they can be used in component assignments, and to deallocate the components
// that may have been allocated in the end.

// RUN: fir-opt --array-value-copy %s | FileCheck %s


!t_with_alloc_comp = type !fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>
func.func private @custom_assign(!fir.ref<!t_with_alloc_comp>, !fir.ref<!t_with_alloc_comp>)
func.func @test_overlap_with_alloc_components(%arg0: !fir.ref<!fir.array<10x!t_with_alloc_comp>>) {
  %0 = fir.alloca !fir.box<!t_with_alloc_comp>
  %c10 = arith.constant 10 : index
  %c9 = arith.constant 9 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  %c0 = arith.constant 0 : index
  %1 = fir.shape %c10 : (index) -> !fir.shape<1>
  %6 = fir.slice %c10, %c1, %c-1 : (index, index, index) -> !fir.slice<1>
  %2 = fir.array_load %arg0(%1) : (!fir.ref<!fir.array<10x!t_with_alloc_comp>>, !fir.shape<1>) -> !fir.array<10x!t_with_alloc_comp>
  %7 = fir.array_load %arg0(%1) [%6] : (!fir.ref<!fir.array<10x!t_with_alloc_comp>>, !fir.shape<1>, !fir.slice<1>) -> !fir.array<10x!t_with_alloc_comp>
  %9 = fir.do_loop %arg1 = %c0 to %c9 step %c1 unordered iter_args(%arg2 = %2) -> (!fir.array<10x!t_with_alloc_comp>) {
    %10 = fir.array_access %7, %arg1 : (!fir.array<10x!t_with_alloc_comp>, index) -> !fir.ref<!t_with_alloc_comp>
    %11 = fir.array_access %arg2, %arg1 : (!fir.array<10x!t_with_alloc_comp>, index) -> !fir.ref<!t_with_alloc_comp>
    fir.call @custom_assign(%11, %10) : (!fir.ref<!t_with_alloc_comp>, !fir.ref<!t_with_alloc_comp>) -> none
    %19 = fir.array_amend %arg2, %11 : (!fir.array<10x!t_with_alloc_comp>, !fir.ref<!t_with_alloc_comp>) -> !fir.array<10x!t_with_alloc_comp>
    fir.result %19 : !fir.array<10x!t_with_alloc_comp>
  }
  fir.array_merge_store %2, %9 to %arg0 : !fir.array<10x!t_with_alloc_comp>, !fir.array<10x!t_with_alloc_comp>, !fir.ref<!fir.array<10x!t_with_alloc_comp>>
  return
}

// CHECK-LABEL: func @test_overlap_with_alloc_components(
// CHECK-SAME:  %[[VAL_0:.*]]: !fir.ref<!fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>) {
// CHECK:   %[[VAL_4:.*]] = arith.constant 10 : index
// CHECK:   %[[VAL_6:.*]] = arith.constant 1 : index
// CHECK:   %[[VAL_7:.*]] = arith.constant -1 : index
// CHECK:   %[[VAL_9:.*]] = fir.shape %[[VAL_4]] : (index) -> !fir.shape<1>
// CHECK:   %[[VAL_10:.*]] = fir.slice %[[VAL_4]], %[[VAL_6]], %[[VAL_7]] : (index, index, index) -> !fir.slice<1>
// CHECK:   %[[VAL_11:.*]] = fir.allocmem !fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>
// CHECK:   %[[VAL_12:.*]] = fir.embox %[[VAL_11]](%[[VAL_9]]) : (!fir.heap<!fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>, !fir.shape<1>) -> !fir.box<!fir.ref<!fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>
// CHECK:   %[[VAL_16:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.ref<!fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>) -> !fir.box<none>
// CHECK:   fir.call @_FortranAInitialize(%[[VAL_16]], %{{.*}}, %{{.*}}) : (!fir.box<none>, !fir.ref<i8>, i32) -> none
// CHECK:   fir.do_loop {{.*}} {
// CHECK:     fir.call @_FortranAAssign
// CHECK:   }
// CHECK:   fir.do_loop {{.*}} {
// CHECK:     fir.call @custom_assign
// CHECK:   }
// CHECK:   fir.do_loop %{{.*}} {
// CHECK:     fir.call @_FortranAAssign
// CHECK:   }
// CHECK:   %[[VAL_72:.*]] = fir.convert %[[VAL_12]] : (!fir.box<!fir.ref<!fir.array<10x!fir.type<t{i:!fir.box<!fir.heap<!fir.array<?xi32>>>}>>>>) -> !fir.box<none>
// CHECK:   %[[VAL_73:.*]] = fir.call @_FortranADestroy(%[[VAL_72]]) : (!fir.box<none>) -> none
// CHECK:   fir.freemem %[[VAL_11]]
