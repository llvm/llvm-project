! Test delayed privatization for pointers: `private`.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 |\
! RUN:   FileCheck %s

subroutine delayed_privatization_pointer
  implicit none
  integer, pointer :: var1

!$omp parallel firstprivate(var1)
  var1 = 10
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.ref<!fir.box<!fir.ptr<i32>>>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:   %[[PRIV_ALLOC:.*]] = fir.alloca !fir.box<!fir.ptr<i32>> {bindc_name = "var1", pinned, uniq_name = "_QFdelayed_privatization_pointerEvar1"}
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : [[TYPE]])

! CHECK-NEXT: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):
! CHECK-NEXT:    %[[ORIG_BASE_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]]
 ! CHECK-NEXT:   fir.store %[[ORIG_BASE_VAL]] to %[[PRIV_PRIV_ARG]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : [[TYPE]])
! CHECK-NEXT: }
