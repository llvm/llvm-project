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
! CHECK-LABEL:  func.func @_QMcopyin_selectorPfoo(
! CHECK:    fir.select_type
! CHECK:    %[[COPYIN:.*]] = acc.copyin varPtr(%{{.*}} : !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) name("x") -> !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>
! CHECK:    acc.parallel dataOperands(%[[COPYIN]] : !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) {
! CHECK:      %[[DECL:.*]]:2 = hlfir.declare %[[COPYIN]] uniq_name("_QMcopyin_selectorFfooEx") : (!fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>) -> (!fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>, !fir.ref<!fir.type<_QMcopyin_selectorTt{i:i32}>>)
! CHECK:      fir.call @_QPbar(%[[DECL]]#0)

! The associate name of an allocatable or pointer selector is described by a
! box whose type still carries fir.heap or fir.ptr, while the data operation
! uses the declared variable box that does not.
subroutine type_is_allocatable(x)
  class(*), allocatable :: x(:,:)
  select type(y => x)
    type is (complex)
      !$acc parallel copy(y)
        y(1,1) = (3,4)
      !$acc end parallel
  end select
end subroutine
! CHECK-LABEL:  func.func @_QMcopyin_selectorPtype_is_allocatable(
! CHECK:    %[[HOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHIFT:.*]]) uniq_name("[[NAME:.*Ey]]") : ([[HOST_TY:!fir.box<!fir.heap<[^,]*>]], {{.*}}) -> ([[ACC_TY:!fir.box<!fir.array<[^,]*>]], {{.*}})
! CHECK:    %[[COPYIN:.*]] = acc.copyin var(%[[HOST]]#0 : [[ACC_TY]]) {{.*}} -> [[ACC_TY]]
! CHECK:    acc.parallel dataOperands(%[[COPYIN]] : [[ACC_TY]]) {
! CHECK:      hlfir.declare %[[COPYIN]](%[[SHIFT]]) skip_rebox uniq_name("[[NAME]]") : ([[ACC_TY]], {{.*}}) -> ([[ACC_TY]], [[ACC_TY]])

subroutine type_is_pointer(x)
  class(*), pointer :: x(:)
  select type(x)
    type is (integer)
      !$acc data copyin(x)
      !$acc parallel loop
      do i = 1, 4
        x(i) = i
      end do
      !$acc end data
  end select
end subroutine
! CHECK-LABEL:  func.func @_QMcopyin_selectorPtype_is_pointer(
! CHECK:    %[[HOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHIFT:.*]]) uniq_name("[[NAME:.*Ex]]") {{.*}} : ([[HOST_TY:!fir.box<!fir.ptr<[^,]*>]], {{.*}}) -> ([[ACC_TY:!fir.box<!fir.array<[^,]*>]], {{.*}})
! CHECK:    %[[COPYIN:.*]] = acc.copyin var(%[[HOST]]#0 : [[ACC_TY]]) {{.*}} -> [[ACC_TY]]
! CHECK:    acc.data dataOperands(%[[COPYIN]] : [[ACC_TY]]) {
! CHECK:      hlfir.declare %[[COPYIN]](%[[SHIFT]]) skip_rebox uniq_name("[[NAME]]") {{.*}} : ([[ACC_TY]], {{.*}}) -> ([[ACC_TY]], [[ACC_TY]])

subroutine class_is_allocatable(x)
  class(t), allocatable :: x(:)
  select type(y => x)
    class is (t)
      !$acc parallel copy(y)
        y(1)%i = 1
      !$acc end parallel
  end select
end subroutine
! CHECK-LABEL:  func.func @_QMcopyin_selectorPclass_is_allocatable(
! CHECK:    %[[HOST:.*]]:2 = hlfir.declare %{{.*}}(%[[SHIFT:.*]]) uniq_name("[[NAME:.*Ey]]") : ([[HOST_TY:!fir.class<!fir.heap<[^,]*>]], {{.*}}) -> ([[ACC_TY:!fir.class<!fir.array<[^,]*>]], {{.*}})
! CHECK:    %[[COPYIN:.*]] = acc.copyin var(%[[HOST]]#0 : [[ACC_TY]]) {{.*}} -> [[ACC_TY]]
! CHECK:    acc.parallel dataOperands(%[[COPYIN]] : [[ACC_TY]]) {
! CHECK:      hlfir.declare %[[COPYIN]](%[[SHIFT]]) skip_rebox uniq_name("[[NAME]]") : ([[ACC_TY]], {{.*}}) -> ([[ACC_TY]], [[ACC_TY]])
end module
