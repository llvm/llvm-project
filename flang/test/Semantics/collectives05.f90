! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in co_reduce subroutine calls based on
! the co_reduce interface defined in section 16.9.49 of the Fortran 2018 standard.
! To Do: add co_reduce to the list of intrinsics

module foo_m
  implicit none

  type foo_t
    integer :: n=0
  contains
    procedure :: derived_type_op
    generic :: operator(+) => derived_type_op
  end type

contains

  pure function derived_type_op(lhs, rhs) result(lhs_op_rhs)
    class(foo_t), intent(in) :: lhs, rhs
    type(foo_t) lhs_op_rhs
    lhs_op_rhs%n = lhs%n + rhs%n
  end function

end module foo_m

program main
  use foo_m, only : foo_t
  implicit none

  type(foo_t) foo
  class(foo_t), allocatable :: polymorphic
  integer i, status, integer_array(1)
  real x
  real vector(1)
  real array(1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1)
  character(len=1) string, message, character_array(1)
  integer coindexed[*]
  logical bool

  ! correct calls, should produce no errors
  call co_reduce(i,      int_op)
  call co_reduce(i,      int_op,                            status)
  call co_reduce(i,      int_op,                            stat=status)
  call co_reduce(i,      int_op,                                         errmsg=message)
  call co_reduce(i,      int_op,                            stat=status, errmsg=message)
  call co_reduce(i,      int_op,            result_image=1, stat=status, errmsg=message)
  call co_reduce(i,      operation=int_op,  result_image=1, stat=status, errmsg=message)
  call co_reduce(a=i,    operation=int_op,  result_image=1, stat=status, errmsg=message)
  call co_reduce(array,  operation=real_op, result_image=1, stat=status, errmsg=message)
  call co_reduce(vector, operation=real_op, result_image=1, stat=status, errmsg=message)
  call co_reduce(string, operation=char_op, result_image=1, stat=status, errmsg=message)
  call co_reduce(foo,    operation=left,    result_image=1, stat=status, errmsg=message)

  allocate(foo_t :: polymorphic)

  ! Test all statically verifiable semantic requirements on co_reduce arguments
  ! Note: We cannot check requirements that relate to "corresponding references." 
  ! References can correspond only if they execute on differing images.  A code that
  ! executes in a single image might be standard-conforming even if the same code
  ! executing in multiple images is not.

  ! argument 'a' cannot be polymorphic
  !ERROR: to be determined
  call co_reduce(polymorphic, derived_type_op)

  ! argument 'a' cannot be coindexed
  !ERROR: (message to be determined)
  call co_reduce(coindexed[1], int_op)

  ! argument 'a' is intent(inout)
  !ERROR: (message to be determined)
  call co_reduce(i + 1, int_op)

  ! operation must be a pure function
  !ERROR: (message to be determined)
  call co_reduce(i, operation=not_pure)

  ! operation must have exactly two arguments
  !ERROR: (message to be determined)
  call co_reduce(i, too_many_args)

  ! operation result must be a scalar
  !ERROR: (message to be determined)
  call co_reduce(i, array_result)

  ! operation result must be non-allocatable
  !ERROR: (message to be determined)
  call co_reduce(i, allocatable_result)

  ! operation result must be non-pointer
  !ERROR: (message to be determined)
  call co_reduce(i, pointer_result)

  ! operation's arguments must be scalars
  !ERROR: (message to be determined)
  call co_reduce(i, array_args)

  ! operation arguments must be non-allocatable
  !ERROR: (message to be determined)
  call co_reduce(i, allocatable_args)

  ! operation arguments must be non-pointer
  !ERROR: (message to be determined)
  call co_reduce(i, pointer_args)

  ! operation arguments must be non-polymorphic
  !ERROR: (message to be determined)
  call co_reduce(i, polymorphic_args)

  ! operation: type of 'operation' result and arguments must match type of argument 'a'
  !ERROR: (message to be determined)
  call co_reduce(i, real_op)

  ! operation: kind type parameter of 'operation' result and arguments must match kind type parameter of argument 'a'
  !ERROR: (message to be determined)
  call co_reduce(x, double_precision_op)

  ! arguments must be non-optional
  !ERROR: (message to be determined)
  call co_reduce(i, optional_args)

  ! if one argument is asynchronous, the other must be also
  !ERROR: (message to be determined)
  call co_reduce(i, asynchronous_mismatch)

  ! if one argument is a target, the other must be also
  !ERROR: (message to be determined)
  call co_reduce(i, target_mismatch)

  ! if one argument has the value attribute, the other must have it also
  !ERROR: (message to be determined)
  call co_reduce(i, value_mismatch)

  ! result_image argument must be an integer scalar
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=integer_array)

  ! result_image argument must be an integer
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=bool)

  ! stat not allowed to be coindexed
  !ERROR: to be determined
  call co_reduce(i, int_op, stat=coindexed[1])

  ! stat argument must be an integer scalar
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=integer_array)

  ! stat argument has incorrect type
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_reduce(i, int_op, result_image=1, string)

  ! stat argument is intent(out)
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=1+1)

  ! errmsg argument must not be coindexed
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=status, errmsg=conindexed_string[1])

  ! errmsg argument must be a scalar
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=status, errmsg=character_array)

  ! errmsg argument must be a character
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=status, errmsg=i)

  ! errmsg argument is intent(inout)
  !ERROR: to be determined
  call co_reduce(i, int_op, result_image=1, stat=status, errmsg="literal constant")

  ! too many arguments to the co_reduce() call
  !ERROR: too many actual arguments for intrinsic 'co_reduce'
  call co_reduce(i, int_op, result_image=1, stat=status, errmsg=message, 3.4)

  ! non-existent keyword argument
  !ERROR: unknown keyword argument to intrinsic 'co_reduce'
  call co_reduce(fake=3.4)

contains

  pure function left(lhs, rhs) result(lhs_op_rhs)
    type(foo_t), intent(in)  :: lhs, rhs
    type(foo_t) :: lhs_op_rhs
    lhs_op_rhs = lhs
  end function

  pure function char_op(lhs, rhs) result(lhs_op_rhs)
    character(len=1), intent(in)  :: lhs, rhs
    character(len=1) :: lhs_op_rhs
    lhs_op_rhs = min(lhs, rhs)
  end function

  pure function real_op(lhs, rhs) result(lhs_op_rhs)
    real, intent(in) :: lhs, rhs
    real :: lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function double_precision_op(lhs, rhs) result(lhs_op_rhs)
    integer, parameter :: double = kind(1.0D0)
    real(double), intent(in) :: lhs, rhs
    real(double) lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function int_op(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in) :: lhs, rhs
    integer :: lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  function not_pure(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in) :: lhs, rhs
    integer :: lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function too_many_args(lhs, rhs, foo) result(lhs_op_rhs)
    integer, intent(in) :: lhs, rhs, foo
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function array_result(lhs, rhs)
    integer, intent(in) :: lhs, rhs
    integer array_result(1)
    array_result = lhs + rhs
  end function

  pure function allocatable_result(lhs, rhs)
    integer, intent(in) :: lhs, rhs
    integer, allocatable :: allocatable_result
    allocatable_result = lhs + rhs
  end function

  pure function pointer_result(lhs, rhs)
    integer, intent(in) :: lhs, rhs
    integer, pointer :: pointer_result
    allocate(pointer_result, source=lhs + rhs )
  end function

  pure function array_args(lhs, rhs)
    integer, intent(in) :: lhs(1), rhs(1)
    integer array_args
    array_args = lhs(1) + rhs(1)
  end function

  pure function allocatable_args(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), allocatable :: lhs, rhs
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function pointer_args(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), pointer :: lhs, rhs
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function polymorphic_args(lhs, rhs) result(lhs_op_rhs)
    class(foo_t), intent(in) :: lhs, rhs
    type(foo_t) lhs_op_rhs
    lhs_op_rhs%n = lhs%n + rhs%n
  end function

  pure function optional_args(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), optional :: lhs, rhs
    integer lhs_op_rhs
    if (present(lhs) .and. present(rhs)) then
      lhs_op_rhs = lhs + rhs
    else
      lhs_op_rhs = 0
    end if
  end function

  pure function target_mismatch(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), target  :: lhs
    integer, intent(in) :: rhs
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function value_mismatch(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), value:: lhs
    integer, intent(in) :: rhs
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

  pure function asynchronous_mismatch(lhs, rhs) result(lhs_op_rhs)
    integer, intent(in), asynchronous:: lhs
    integer, intent(in) :: rhs
    integer lhs_op_rhs
    lhs_op_rhs = lhs + rhs
  end function

end program
