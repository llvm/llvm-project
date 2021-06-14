! RUN: %S/test_errors.sh %s %t %flang_fc1
! REQUIRES: shell
! Test various conditions in C1158.
implicit none

type :: t1
  integer :: i
end type

type, extends(t1) :: t2
end type

type(t1),target :: x1
type(t2),target :: x2

class(*), pointer :: ptr
class(t1), pointer :: p_or_c
!vector subscript related
class(t1),DIMENSION(:,:),allocatable::array1
class(t2),DIMENSION(:,:),allocatable::array2
integer, dimension(2) :: V
V = (/ 1,2 /)
allocate(array1(3,3))
allocate(array2(3,3))

! A) associate with function, i.e (other than variables)
select type ( y => fun(1) )
  type is (t1)
    print *, rank(y%i)
end select

select type ( y => fun(1) )
  type is (t1)
    !ERROR: Left-hand side of assignment is not modifiable
    y%i = 1 !VDC
  type is (t2)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'z=' must be definable
    call sub_with_in_and_inout_param(y,y) !VDC
end select

! B) associated with a variable:
p_or_c => x1
select type ( a => p_or_c )
  type is (t1)
    a%i = 10
end select

select type ( a => p_or_c )
  type is (t1)
end select

!C)Associate with  with vector subscript
select type (b => array1(V,2))
  type is (t1)
    !ERROR: Left-hand side of assignment is not modifiable
    b%i  = 1 !VDC
  type is (t2)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'z=' must be definable
    call sub_with_in_and_inout_param_vector(b,b) !VDC
end select
select type(b =>  foo(1) )
  type is (t1)
    !ERROR: Left-hand side of assignment is not modifiable
    b%i = 1 !VDC
  type is (t2)
    !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'z=' must be definable
    call sub_with_in_and_inout_param_vector(b,b) !VDC
end select

!D) Have no association and should be ok.
!1. points to function
ptr => fun(1)
select type ( ptr )
type is (t1)
  ptr%i = 1
end select

!2. points to variable
ptr=>x1
select type (ptr)
  type is (t1)
    ptr%i = 10
end select

contains

  function fun(i)
    class(t1),pointer :: fun
    integer :: i
    if (i>0) then
      fun => x1
    else if (i<0) then
      fun => x2
    else
      fun => NULL()
    end if
  end function

  function foo(i)
    integer :: i
    class(t1),DIMENSION(:),allocatable :: foo
    integer, dimension(2) :: U
    U = (/ 1,2 /)  
    if (i>0) then
      foo = array1(2,U)
    else if (i<0) then
      !ERROR: No intrinsic or user-defined ASSIGNMENT(=) matches operand types TYPE(t1) and TYPE(t2)
      foo = array2(2,U)
    end if
  end function

  subroutine sub_with_in_and_inout_param(y, z)
    type(t2), INTENT(IN) :: y
    class(t2), INTENT(INOUT) :: z
    z%i = 10
  end subroutine

  subroutine sub_with_in_and_inout_param_vector(y, z)
    type(t2),DIMENSION(:), INTENT(IN) :: y
    class(t2),DIMENSION(:), INTENT(INOUT) :: z
    z%i = 10
  end subroutine

end
