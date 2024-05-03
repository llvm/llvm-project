! Test that parent components are made explicit in reference to
! procedure pointer from parent type.
! RUN: bbc -emit-hlfir -o - %s | FileCheck %s

module type_defs
 interface
  subroutine s1
  end subroutine
  real function s2()
  end function
 end interface
 type :: t
  procedure(s1), pointer, nopass :: p1
  procedure(s2), pointer, nopass :: p2
 end type
 type, extends(t) :: t2
 end type
end module

! CHECK-LABEL: func.func @_QPtest(
subroutine test (x)
use type_defs, only : t2
type(t2) :: x
call x%p1()
! CHECK: %[[T_REF1:.*]] = hlfir.designate %{{.*}}{"t"}
! CHECK: hlfir.designate %[[T_REF1]]{"p1"}
print *, x%p2()
! CHECK: %[[T_REF2:.*]] = hlfir.designate %{{.*}}{"t"}
! CHECK: hlfir.designate %[[T_REF2]]{"p2"}
end subroutine
