! Test HLFIR lowering of user defined elemental procedure references
! with finalizable results. Verify that the elemental results
! are not destroyed inside hlfir.elemental.
! RUN: bbc -emit-hlfir -o - -I nowhere %s 2>&1 | FileCheck %s

module types
  type t
   contains
     final :: finalize
  end type t
contains
  pure subroutine finalize(x)
    type(t), intent(inout) :: x
  end subroutine finalize
end module types

subroutine test1(x)
  use types
  interface
     elemental function elem(x)
       use types
       type(t), intent(in) :: x
       type(t) :: elem
     end function elem
  end interface
  type(t) :: x(:)
  x = elem(x)
end subroutine test1
! CHECK-LABEL:   func.func @_QPtest1(
! CHECK:           %[[VAL_6:.*]] = hlfir.elemental %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>> {
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK:           hlfir.destroy %[[VAL_6]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>

subroutine test2(x)
  use types
  interface
     elemental function elem(x)
       use types
       type(t), intent(in) :: x
       type(t) :: elem
     end function elem
     elemental function elem2(x, y)
       use types
       type(t), intent(in) :: x, y
       type(t) :: elem2
     end function elem2
  end interface
  type(t) :: x(:)
  x = elem2(elem(x), elem(x))
end subroutine test2
! CHECK-LABEL:   func.func @_QPtest2(
! CHECK:           %[[VAL_8:.*]] = hlfir.elemental %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>> {
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK:           %[[VAL_16:.*]] = hlfir.elemental %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>> {
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK:           %[[VAL_23:.*]] = hlfir.elemental %{{.*}} : (!fir.shape<1>) -> !hlfir.expr<?x!fir.type<_QMtypesTt>> {
! CHECK-NOT: fir.call @_FortranADestroy
! CHECK:           hlfir.destroy %[[VAL_23]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>
! CHECK:           hlfir.destroy %[[VAL_16]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>
! CHECK:           hlfir.destroy %[[VAL_8]] finalize : !hlfir.expr<?x!fir.type<_QMtypesTt>>
