! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! Check for semantic errors in co_sum() subroutine calls
! To Do: add co_sum to the evaluation stage

module test_co_sum
  implicit none

contains

  subroutine test
  
    integer i, status
    real array(1)
    complex z(1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1) 
    character(len=1) message
  
    ! correct calls, should produce no errors
    call co_sum(i)
    call co_sum(i,   1)
    call co_sum(i,   1,              status)
    call co_sum(i,   1,              stat=status)
    call co_sum(i,   1,                           errmsg=message)
    call co_sum(i,   1,              stat=status, errmsg=message)
    call co_sum(i,   result_image=1, stat=status, errmsg=message)
    call co_sum(a=i, result_image=1, stat=status, errmsg=message)
    call co_sum(i,   result_image=1, stat=status, errmsg=message)
  
    call co_sum(array, result_image=1, stat=status, errmsg=message)
    call co_sum(z, result_image=1, stat=status, errmsg=message)
 
    ! the error is seen as an incorrect type for the stat= argument
    !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
    call co_sum(i, 1, message)
 
    ! the error is seen as too many arguments to the co_sum() call
    !ERROR: too many actual arguments for intrinsic 'co_sum'
    call co_sum(i,   result_image=1, stat=status, errmsg=message, 3.4)
  
    ! keyword argument with incorrect type
    !ERROR: unknown keyword argument to intrinsic 'co_sum'
    call co_sum(fake=3.4)
  
  end subroutine

end module test_co_sum
