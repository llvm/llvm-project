! Test delayed privatization for allocatables: `private`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_allocatable
  implicit none
  integer, allocatable :: var1

!$omp parallel private(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = private}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_allocatableEvar1"}

! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]] : (!fir.heap<i32>) -> i64
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi ne, %[[PRIV_ARG_ADDR]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK:          %[[PRIV_ALLOCMEM:.*]] = fir.allocmem i32 {fir.must_be_heap = true, uniq_name = "_QFdelayed_privatization_allocatableEvar1.alloc"}
! CHECK-NEXT:     %[[PRIV_ALLOCMEM_BOX:.*]] = fir.embox %[[PRIV_ALLOCMEM]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[PRIV_ALLOCMEM_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[ZERO_BITS:.*]] = fir.zero_bits !fir.heap<i32>
! CHECK-NEXT:     %[[ZERO_BOX:.*]] = fir.embox %[[ZERO_BITS]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK-NEXT:     fir.store %[[ZERO_BOX]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK-NEXT:   }

! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : [[TYPE]])

! CHECK-NEXT: } dealloc {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ADDR:.*]] = fir.box_addr %[[PRIV_VAL]]
! CHECK-NEXT:   %[[PRIV_ADDR_I64:.*]] = fir.convert %[[PRIV_ADDR]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[PRIV_NULL_COND:.*]] = arith.cmpi ne, %[[PRIV_ADDR_I64]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[PRIV_NULL_COND]] {
! CHECK:          %[[PRIV_VAL_2:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:     %[[PRIV_ADDR_2:.*]] = fir.box_addr %[[PRIV_VAL_2]]
! CHECK-NEXT:     fir.freemem %[[PRIV_ADDR_2]]
! CHECK-NEXT:     %[[ZEROS:.*]] = fir.zero_bits
! CHECK-NEXT:     %[[ZEROS_BOX:.*]]  = fir.embox %[[ZEROS]]
! CHECK-NEXT:     fir.store %[[ZEROS_BOX]] to %[[PRIV_ARG]]
! CHECK-NEXT:   }

! CHECK-NEXT:   omp.yield
