! RUN: %flang_fc1 -emit-hlfir -fcoarray %s -o - | FileCheck %s

subroutine test_coarray_cleanup()
    implicit none
    real, allocatable :: n(:)[:]
    
    allocate(n(10)[*])
    
    sync all
end subroutine test_coarray_cleanup

!CHECK:  %[[VAL_0:.*]] = fir.alloca !fir.array<0xi64>
!CHECK:  %[[VAL_1:.*]] = fir.alloca !fir.array<1xi64>
!CHECK:  %[[VAL_2:.*]] = fir.dummy_scope : !fir.dscope
!CHECK:  %[[VAL_3:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>> {bindc_name = "n", uniq_name = "_QFtest_coarray_cleanupEn"}
!CHECK:  %[[VAL_4:.*]] = fir.zero_bits !fir.heap<!fir.array<?xf32>>
!CHECK:  %[[C0:.*]] = arith.constant 0 : index
!CHECK:  %[[VAL_5:.*]] = fir.shape %[[C0]] : (index) -> !fir.shape<1>
!CHECK:  %[[VAL_6:.*]] = fir.embox %[[VAL_4]](%[[VAL_5]]) : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
!CHECK:  fir.store %[[VAL_6]] to %[[VAL_3]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK:  %[[VAL_7:.*]]:2 = hlfir.declare %[[VAL_3]] {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtest_coarray_cleanupEn"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
!CHECK:  %[[VAL_8:.*]] = fir.absent !fir.box<none>
!CHECK:  %[[C1:.*]] = arith.constant 1 : index
!CHECK:  %[[C10_i32:.*]] = arith.constant 10 : i32
!CHECK:  %[[C0_i32:.*]] = arith.constant 0 : i32
!CHECK:  %[[VAL_9:.*]] = fir.convert %[[VAL_7]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<none>>
!CHECK:  %[[VAL_10:.*]] = fir.convert %[[C1]] : (index) -> i64
!CHECK:  %[[VAL_11:.*]] = fir.convert %[[C10_i32]] : (i32) -> i64
!CHECK:  fir.call @_FortranAAllocatableSetBounds(%[[VAL_9:.*]], %[[C0_i32]], %[[VAL_10]], %[[VAL_11]]) fastmath<contract> : (!fir.ref<!fir.box<none>>, i32, i64, i64) -> ()
!CHECK:  %[[C1_i64:.*]] = arith.constant 1 : i64
!CHECK:  %[[C0_0:.*]] = arith.constant 0 : index
!CHECK:  %[[VAL_12:.*]] = fir.coordinate_of %[[VAL_1]], %[[C0_0]] : (!fir.ref<!fir.array<1xi64>>, index) -> !fir.ref<i64>
!CHECK:  fir.store %[[C1_i64]] to %[[VAL_12]] : !fir.ref<i64>
!CHECK:  %[[VAL_13:.*]] = fir.embox %[[VAL_1]] : (!fir.ref<!fir.array<1xi64>>) -> !fir.box<!fir.array<1xi64>>
!CHECK:  %[[VAL_14:.*]] = fir.embox %[[VAL_0]] : (!fir.ref<!fir.array<0xi64>>) -> !fir.box<!fir.array<0xi64>>
!CHECK:  mif.alloc_coarray %[[VAL_7]]#0 lcobounds %[[VAL_13]] ucobounds %[[VAL_14]] errmsg %[[VAL_8]] {uniq_name = "_QFtest_coarray_cleanupEn"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.box<!fir.array<1xi64>>, !fir.box<!fir.array<0xi64>>, !fir.box<none>) -> ()
!CHECK:  mif.sync_all : () -> ()
!CHECK:  %[[VAL_15:.*]] = fir.load %[[VAL_7]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
!CHECK:  %[[VAL_16:.*]] = fir.box_addr %[[VAL_15]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
!CHECK:  %[[VAL_17:.*]] = fir.convert %[[VAL_16]] : (!fir.heap<!fir.array<?xf32>>) -> i64
!CHECK:  %[[C0_i64:.*]] = arith.constant 0 : i64
!CHECK:  %[[VAL_18:.*]] = arith.cmpi ne, %[[VAL_17]], %[[C0_i64]] : i64
!CHECK:  fir.if %[[VAL_18]] {
!CHECK:    %[[VAL_19:.*]] = fir.absent !fir.box<none>
!CHECK:    %[[VAL_20:.*]] = fir.absent !fir.ref<i32>
!CHECK:    mif.dealloc_coarray %[[VAL_7]]#0 stat %[[VAL_20]] errmsg %[[VAL_19]] : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<i32>, !fir.box<none>) -> ()
!CHECK:  }
!CHECK:  return

