! Test remapping of select type selector.
! RUN: %flang_fc1 -fopenacc -emit-hlfir %s -o - | FileCheck %s

module copyin_selector
  type t
    integer :: i
  end type
contains
subroutine foo(x)
  class(*) :: x
  select type(x)
    type is (t)
      !$acc parallel copyin(x)
        call bar(x)
      !$acc end parallel
  end select
end subroutine
end module
! CHECK-LABEL:  func.func @_QMcopyin_selectorPfoo(
! CHECK:    fir.select_type
! CHECK:    %[[COPYIN:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) -> !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>> {name = "x"}
! CHECK:    acc.parallel dataOperands(%[[COPYIN]] : !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) {
! CHECK:      %[[DECL:.*]]:2 = hlfir.declare %[[COPYIN]] {uniq_name = "_QMcopyin_selectorFfooEx"} : (!fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) -> (!fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>, !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>)
! CHECK:      fir.call @_QPbar(%[[DECL]]#0)
