! RUN: bbc -emit-fir -polymorphic-type %s -o - | FileCheck %s

module same_type_as_mod

  type p1
    integer :: a
  end type

  type, extends(p1) :: p2
    integer :: b
  end type

  type k1(a)
    integer, kind :: a
  end type

contains
  subroutine is_same_type(a, b)
    class(*) :: a
    class(*) :: b

    if (same_type_as(a, b)) then
      print*, 'same_type_as ok'
    else
      print*, 'same_type_as failed'
    end if
  end subroutine

! CHECK-LABEL: func.func @_QMsame_type_as_modPis_same_type(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.class<none> {fir.bindc_name = "b"}) {
! CHECK: %[[BOX0:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: %[[BOX1:.*]] = fir.convert %[[ARG1]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranASameTypeAs(%[[BOX0]], %[[BOX1]]) {{.*}} : (!fir.box<none>, !fir.box<none>) -> i1

end module

program test
  use same_type_as_mod
  type(p1) :: p, r
  type(p2) :: q
  type(k1(10)) :: k10
  type(k1(20)) :: k20

  call is_same_type(p, q)
  call is_same_type(p, r)
  call is_same_type(k10, k20)
end
