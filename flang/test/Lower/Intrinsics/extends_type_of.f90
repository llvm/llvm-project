! RUN: bbc -emit-fir -hlfir=false %s -o - | FileCheck %s

module extends_type_of_mod

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
  subroutine is_extended_type(a, b)
    class(*) :: a
    class(*) :: b

    if (extends_type_of(a, b)) then
      print*, 'extends_type_of ok'
    else
      print*, 'extends_type_of failed'
    end if
  end subroutine

! CHECK-LABEL: func.func @_QMextends_type_of_modPis_extended_type(
! CHECK-SAME: %[[ARG0:.*]]: !fir.class<none> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.class<none> {fir.bindc_name = "b"}) {
! CHECK: %[[BOX0:.*]] = fir.convert %[[ARG0]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: %[[BOX1:.*]] = fir.convert %[[ARG1]] : (!fir.class<none>) -> !fir.box<none>
! CHECK: %{{.*}} = fir.call @_FortranAExtendsTypeOf(%[[BOX0]], %[[BOX1]]) {{.*}} : (!fir.box<none>, !fir.box<none>) -> i1

end module

program test
  use extends_type_of_mod
  type(p1) :: p, r
  type(p2) :: q
  type(k1(10)) :: k10
  type(k1(20)) :: k20

  call is_extended_type(p, p)
  call is_extended_type(p, q)
  call is_extended_type(p, r)
  call is_extended_type(q, p)
  call is_extended_type(k10, k20)
end
