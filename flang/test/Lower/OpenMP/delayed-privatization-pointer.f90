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

subroutine delayed_privatization_lenparams(length)
  integer, intent(in) :: length
  character(length), pointer :: var

  !$omp parallel firstprivate(var)
    var = 'a'
  !$omp end parallel
end subroutine

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM2:.*]] : [[TYPE:!fir.box<!fir.ptr<!fir.char<1,\?>>>]] init {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<[[TYPE]]>, %[[PRIV_ALLOC:.*]]: !fir.ref<[[TYPE]]>):
! CHECK-NEXT:   %[[ARG:.*]] = fir.load %[[PRIV_ARG]]
! CHECK-NEXT:   %[[SIZE:.*]] = fir.box_elesize %[[ARG]]
! CHECK-NEXT:   %[[NULL:.*]] = fir.zero_bits !fir.ptr<!fir.char<1,?>>
! CHECK-NEXT:   %[[INIT:.*]] = fir.embox %[[NULL]] typeparams %[[SIZE]]
! CHECK-NEXT:   fir.store %[[INIT]] to %[[PRIV_ALLOC]]
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<[[TYPE]]>)
! CHECK-NEXT: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<[[TYPE]]>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<[[TYPE]]>):
! CHECK-NEXT:    %[[ORIG_BASE_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]]
! CHECK-NEXT:   fir.store %[[ORIG_BASE_VAL]] to %[[PRIV_PRIV_ARG]]
! CHECK-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<[[TYPE]]>)
! CHECK-NEXT: }

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME: @[[PRIVATIZER_SYM:.*]] : [[TYPE:!fir.box<!fir.ptr<i32>>]] init {
! CHECK-NEXT: ^bb0(%[[PRIV_ARG:.*]]: !fir.ref<[[TYPE]]>, %[[PRIV_ALLOC:.*]]: !fir.ref<[[TYPE]]>):
! CHECK-NEXT:   %[[NULL:.*]] = fir.zero_bits !fir.ptr<i32>
! CHECK-NEXT:   %[[INIT:.*]] = fir.embox %[[NULL]] : (!fir.ptr<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK-NEXT:   fir.store %[[INIT]] to %[[PRIV_ALLOC]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:   omp.yield(%[[PRIV_ALLOC]] : !fir.ref<[[TYPE]]>)
! CHECK-NEXT: } copy {
! CHECK: ^bb0(%[[PRIV_ORIG_ARG:.*]]: !fir.ref<[[TYPE]]>, %[[PRIV_PRIV_ARG:.*]]: !fir.ref<[[TYPE]]>):
! CHECK-NEXT:    %[[ORIG_BASE_VAL:.*]] = fir.load %[[PRIV_ORIG_ARG]]
! CHECK-NEXT:   fir.store %[[ORIG_BASE_VAL]] to %[[PRIV_PRIV_ARG]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK-NEXT:   omp.yield(%[[PRIV_PRIV_ARG]] : !fir.ref<[[TYPE]]>)
! CHECK-NEXT: }
