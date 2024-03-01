! Test delayed privatization for allocatables.

! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 | FileCheck %s

subroutine delayed_privatization_allocatable
  implicit none
  integer, allocatable :: var1

!$omp parallel firstprivate(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.shadow<!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>, allocatable : true>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:  %[[BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [0 : index] : ([[TYPE]])
! CHECK-NEXT:  %[[FIR_BASE:.*]] = fir.extract_value %[[PRIV_ARG]], [1 : index] : ([[TYPE]])

! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_allocatableEvar1"}

! CHECK-NEXT:   %[[FIR_BASE_VAL:.*]] = fir.load %[[FIR_BASE]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   %[[FIR_BASE_BOX:.*]] = fir.box_addr %[[FIR_BASE_VAL]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK-NEXT:   %[[FIR_BASE_ADDR:.*]] = fir.convert %[[FIR_BASE_BOX]] : (!fir.heap<i32>) -> i64
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi ne, %[[FIR_BASE_ADDR]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK-NEXT:     %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32 {fir.must_be_heap = true, uniq_name = "_QFdelayed_privatization_allocatableEvar1.alloc"}
! CHECK-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[ZERO_BITS]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   }

! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]]

! CHECK-NEXT:   %[[PRIV_SHADOW:.*]] = fir.undefined [[TYPE]]
! CHECK-NEXT:   %[[PRIV_SHADOW_2:.*]] = fir.insert_value %[[PRIV_SHADOW]], %[[PRIV_DECL]]#0, [0 : index]
! CHECK-NEXT:   %[[PRIV_SHADOW_3:.*]] = fir.insert_value %[[PRIV_SHADOW_2]], %[[PRIV_DECL]]#1, [1 : index]

! CHECK-NEXT:   omp.yield(%[[PRIV_SHADOW_3]] : [[TYPE]])

! CHECK-NEXT: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! CHECK-NEXT:  %[[ORIG_BASE:.*]] = fir.extract_value %[[PRIV_ORIG_ARG]], [0 : index] : ([[TYPE]])
! CHECK-NEXT:  %[[ORIG_FIR_BASE:.*]] = fir.extract_value %[[PRIV_ORIG_ARG]], [1 : index] : ([[TYPE]])

! CHECK-NEXT:  %[[PRIV_BASE:.*]] = fir.extract_value %[[PRIV_PRIV_ARG]], [0 : index] : ([[TYPE]])
! CHECK-NEXT:  %[[PRIV_FIR_BASE:.*]] = fir.extract_value %[[PRIV_PRIV_ARG]], [1 : index] : ([[TYPE]])

! CHECK-NEXT:  %[[PRIV_BASE_VAL:.*]] = fir.load %[[PRIV_BASE]]
! CHECK-NEXT:  %[[PRIV_BASE_BOX:.*]] = fir.box_addr %[[PRIV_BASE_VAL]]
! CHECK-NEXT:  %[[PRIV_BASE_ADDR:.*]] = fir.convert %[[PRIV_BASE_BOX]]
! CHECK-NEXT:  %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:  %[[COPY_COND:.*]] = arith.cmpi ne, %[[PRIV_BASE_ADDR]], %[[C0]] : i64

! CHECK-NEXT:  fir.if %[[COPY_COND]] {
! CHECK-NEXT:    %[[ORIG_BASE_VAL:.*]] = fir.load %[[ORIG_BASE]]
! CHECK-NEXT:    %[[ORIG_BASE_ADDR:.*]] = fir.box_addr %[[ORIG_BASE_VAL]]
! CHECK-NEXT:    %[[ORIG_BASE_LD:.*]] = fir.load %[[ORIG_BASE_ADDR]]
! CHECK-NEXT:    hlfir.assign %[[ORIG_BASE_LD]] to %[[PRIV_BASE_BOX]] temporary_lhs
! CHECK-NEXT:  }

! CHECK-NEXT:  omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
