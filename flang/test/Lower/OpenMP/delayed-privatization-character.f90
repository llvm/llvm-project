! Test delayed privatization for the `CHARACTER` type.

! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --openmp-enable-delayed-privatization \
! RUN:   -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --openmp-enable-delayed-privatization -o - %s 2>&1 \
! RUN:   | FileCheck %s

subroutine delayed_privatization_character(var1, l)
  implicit none
  integer(8):: l
  character(len = l)  :: var1

!$omp parallel firstprivate(var1)
  var1 = "test"
!$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.boxchar<1>]] alloc {

! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: [[TYPE]]):
! CHECK-NEXT:   %[[UNBOX:.*]]:2 = fir.unboxchar %[[PRIV_ARG]]
! CHECK:        %[[PRIV_ALLOC:.*]] = fir.alloca !fir.char<1,?>(%[[UNBOX]]#1 : index)
! CHECK-NEXT:   %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ALLOC]] typeparams %[[UNBOX]]#1
! CHECK-NEXT:   omp.yield(%[[PRIV_DECL]]#0 : !fir.boxchar<1>)

! CHECK-NEXT: } copy {
! CHECK-NEXT: ^bb0(%[[PRIV_ORIG_ARG:.*]]: [[TYPE]], %[[PRIV_PRIV_ARG:.*]]: [[TYPE]]):

! CHECK-NEXT:   hlfir.assign %[[PRIV_ORIG_ARG]] to %[[PRIV_PRIV_ARG]]

! CHECK-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : !fir.boxchar<1>)
! CHECK-NEXT: }
