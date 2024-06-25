! Test delayed privatization for allocatable arrays.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_private(var1, l1)
  implicit none
  integer(8):: l1
  integer, allocatable, dimension(:) :: var1

!$omp parallel firstprivate(var1)
  var1(l1 + 1) = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.box<!fir.heap<!fir.array<\?xi32>>>>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<{{\?}}xi32>>> {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_privateEvar1"}

! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]]
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi ne, %[[PRIV_ARG_ADDR]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK-NEXT:     %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]] : [[TYPE]]
! CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:     %[[DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]]
! CHECK-NEXT:     fir.box_addr %[[PRIV_ARG_VAL]]
! CHECK-NEXT:     %[[C0_2:.*]] = arith.constant 0 : index 
! CHECK-NEXT:     %[[CMP:.*]] = arith.cmpi sgt, %[[DIMS]]#1, %[[C0_2]] : index
! CHECK-NEXT:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[DIMS]]#1, %[[C0_2]] : index
! CHECK-NEXT:     %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[SELECT]]
! CHECK-NEXT:     %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS]]#0, %[[SELECT]] : (index, index) -> !fir.shapeshift<1>
! CHECK-NEXT:     %[[EMBOX:.*]] = fir.embox %[[MEM]](%[[SHAPE_SHIFT]])
! CHECK-NEXT:     fir.store %[[EMBOX]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[ZEROS:.*]] = fir.zero_bits
! CHECK-NEXT:     %[[C0_3:.*]] = arith.constant 0 : index
! CHECK-NEXT:     %[[SHAPE:.*]] = fir.shape %[[C0_3]] : (index) -> !fir.shape<1>
! CHECK-NEXT:     %[[EMBOX_2:.*]] = fir.embox %[[ZEROS]](%[[SHAPE]])
! CHECK-NEXT:     fir.store %[[EMBOX_2]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   }

! CHECK-NEXT:   hlfir.declare
! CHECK-NEXT:   omp.yield

! CHECK-NEXT: } copy {
! CHECK-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! CHECK-NEXT:  %[[PRIV_BASE_VAL:.*]] = fir.load %[[PRIV_PRIV_ARG]]
! CHECK-NEXT:  %[[PRIV_BASE_BOX:.*]] = fir.box_addr %[[PRIV_BASE_VAL]]
! CHECK-NEXT:  %[[PRIV_BASE_ADDR:.*]] = fir.convert %[[PRIV_BASE_BOX]]
! CHECK-NEXT:  %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:  %[[COPY_COND:.*]] = arith.cmpi ne, %[[PRIV_BASE_ADDR]], %[[C0]] : i64


! CHECK-NEXT:  fir.if %[[COPY_COND]] {
! CHECK-NEXT:    %[[PRIV_ORIG_ARG_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]]
! CHECK-NEXT:    hlfir.assign %[[PRIV_ORIG_ARG_VAL]] to %[[PRIV_BASE_VAL]] temporary_lhs
! CHECK-NEXT:   }
! CHECK-NEXT:   omp.yield
! CHECK-NEXT: }
