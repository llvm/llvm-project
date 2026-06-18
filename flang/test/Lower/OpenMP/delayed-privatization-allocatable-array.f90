! Test delayed privatization for allocatable arrays.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization -o - %s 2>&1 |\
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
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[BOX_TYPE:!fir.box<!fir.heap<!fir.array<\?xi32>>>]] init {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE:!fir.ref<!fir.box<!fir.heap<!fir.array<\?xi32>>>>]], %[[PRIV_ALLOC:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_ARG_VAL:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[PRIV_ARG_BOX:.*]] = fir.box_addr %[[PRIV_ARG_VAL]]
! CHECK-NEXT:   %[[PRIV_ARG_ADDR:.*]] = fir.convert %[[PRIV_ARG_BOX]]
! CHECK-NEXT:   %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:   %[[ALLOC_COND:.*]] = arith.cmpi eq, %[[PRIV_ARG_ADDR]], %[[C0]] : i64

! CHECK-NEXT:   fir.if %[[ALLOC_COND]] {
! CHECK-NEXT:     %[[C0_2:.*]] = arith.constant 0 : index
! CHECK-NEXT:     %[[SHAPE:.*]] = fir.shape %[[C0_2]]
! CHECK-NEXT:     %[[EMBOX_2:.*]] = fir.embox %[[PRIV_ARG_BOX]](%[[SHAPE]])
! CHECK-NEXT:     fir.store %[[EMBOX_2]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   } else {
! CHECK-NEXT:     %[[C0:.*]] = arith.constant 0 : index
! CHECK-NEXT:     %[[DIMS:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0]]
! CHECK-NEXT:     %[[SHAPE:.*]] = fir.shape %[[DIMS]]#1
! CHECK-NEXT:     %[[MEM:.*]] = fir.allocmem !fir.array<?xi32>, %[[DIMS]]#1
! CHECK-NEXT:     %[[DECL:.*]]:2 = hlfir.declare %[[MEM]](%[[SHAPE]])
! CHECK-NEXT:     %[[C0_2:.*]] = arith.constant 0 : index
! CHECK-NEXT:     %[[DIMS_2:.*]]:3 = fir.box_dims %[[PRIV_ARG_VAL]], %[[C0_2]]
! CHECK-NEXT:     %[[SHAPE_SHIFT:.*]] = fir.shape_shift %[[DIMS_2]]#0, %[[DIMS_2]]#1
! CHECK-NEXT:     %[[EMBOX:.*]] = fir.rebox %[[DECL]]#0(%[[SHAPE_SHIFT]])
! CHECK-NEXT:     fir.store %[[EMBOX]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   }

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
! CHECK-NEXT:    hlfir.assign %[[PRIV_ORIG_ARG_VAL]] to %[[PRIV_PRIV_ARG]] realloc
! CHECK-NEXT:   }
! CHECK-NEXT:   omp.yield
! CHECK-NEXT: }
