! Test delayed privatization for allocatables: `firstprivate`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_allocatable
  implicit none
  integer, allocatable :: var1

!$omp parallel firstprivate(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.box<!fir.heap<i32>>>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:  %[[PRIV_BASE_VAL:.*]] = fir.load %[[PRIV_PRIV_ARG]]
! CHECK-NEXT:  %[[PRIV_BASE_BOX:.*]] = fir.box_addr %[[PRIV_BASE_VAL]]
! CHECK-NEXT:  %[[PRIV_BASE_ADDR:.*]] = fir.convert %[[PRIV_BASE_BOX]]
! CHECK-NEXT:  %[[C0:.*]] = arith.constant 0 : i64
! CHECK-NEXT:  %[[COPY_COND:.*]] = arith.cmpi ne, %[[PRIV_BASE_ADDR]], %[[C0]] : i64

! CHECK-NEXT:  fir.if %[[COPY_COND]] {
! CHECK-NEXT:    %[[ORIG_BASE_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]]
! CHECK-NEXT:    %[[ORIG_BASE_ADDR:.*]] = fir.box_addr %[[ORIG_BASE_VAL]]
! CHECK-NEXT:    %[[ORIG_BASE_LD:.*]] = fir.load %[[ORIG_BASE_ADDR]]
! CHECK-NEXT:    hlfir.assign %[[ORIG_BASE_LD]] to %[[PRIV_BASE_BOX]] temporary_lhs
! CHECK-NEXT:  }
