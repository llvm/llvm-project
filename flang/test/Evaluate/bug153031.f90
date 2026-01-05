! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! Ensure that UBOUND() calculation from LBOUND()+SIZE() isn't applied to
! variables containing references to impure functions.
type t
  real, allocatable :: a(:)
end type
interface
  pure integer function pure(n)
    integer, intent(in) :: n
  end
end interface
type(t) :: x(10)
allocate(x(1)%a(2))
!CHECK: PRINT *, ubound(x(int(impure(1_4),kind=8))%a,dim=1_4)
print *, ubound(x(impure(1))%a, dim=1)
!CHECK: PRINT *, int(size(x(int(pure(1_4),kind=8))%a,dim=1,kind=8)+lbound(x(int(pure(1_4),kind=8))%a,dim=1,kind=8)-1_8,kind=4)
print *, ubound(x(pure(1))%a, dim=1)
end
