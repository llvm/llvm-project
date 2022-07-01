! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in co_max subroutine calls based on
! the co_max interface defined in section 16.9.47 of the Fortran 2018 standard.
! To Do: add co_max to the list of intrinsics

program test_co_max
  implicit none

  integer          i, integer_array(1), coindexed_integer[*], status
  character(len=1) c, character_array(1), coindexed_character[*], message
  double precision d, double_precision_array(1)
  real             r, real_array(1), coindexed_real[*]
  complex          complex_type
  logical          bool
  
  !___ standard-conforming calls with no keyword arguments ___
  call co_max(i)
  call co_max(c)
  call co_max(d)
  call co_max(r)
  call co_max(i, 1)
  call co_max(c, 1, status)
  call co_max(d, 1, status, message)
  call co_max(r, 1, status, message)
  call co_max(integer_array)
  call co_max(character_array, 1)
  call co_max(double_precision_array, 1, status)
  call co_max(real_array, 1, status, message)

  !___ standard-conforming calls with keyword arguments ___

  ! all arguments present
  call co_max(a=i, result_image=1, stat=status, errmsg=message) 
  call co_max(result_image=1, a=i, errmsg=message, stat=status) 

  ! one optional argument not present
  call co_max(a=i,                 stat=status, errmsg=message) 
  call co_max(a=i, result_image=1,              errmsg=message)
  call co_max(a=i, result_image=1, stat=status                )

  ! two optional arguments not present
  call co_max(a=i, result_image=1                             ) 
  call co_max(a=i,                 stat=status                )
  call co_max(a=i,                              errmsg=message) 

  ! no optional arguments present
  call co_max(a=i) 

  !___ non-standard-conforming calls ___

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
  call co_max(bool)
  
  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'COMPLEX(4)'
  call co_max(complex_type)
  
  ! argument 'a' is intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' must be definable
  call co_max(a=1+1)
  
  ! argument 'a' shall not be a coindexed object
  !ERROR: to be determined
  call co_max(a=coindexed_real[1])
  
  ! 'result_image' argument shall be a scalar
  !ERROR: too many actual arguments for intrinsic 'co_max'
  call co_max(i, result_image=bool)
  
  ! 'result_image' argument shall be an integer scalar
  !ERROR: too many actual arguments for intrinsic 'co_max'
  call co_max(c, result_image=integer_array)
  
  ! 'stat' argument shall be noncoindexed
  !ERROR: to be determined
  call co_max(d, stat=coindexed_integer[1])
 
  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_max(r, stat=message)
 
  ! 'stat' argument shall be an integer scalar
  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_max(i, stat=integer_array)
 
  ! 'errmsg' argument shall be intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' must be definable
  call co_max(a=i, result_image=1, stat=status, errmsg='c')

  ! 'errmsg' argument shall be noncoindexed
  !ERROR: to be determined
  call co_max(c, errmsg=coindexed_character[1])

  ! 'errmsg' argument shall be a character
  !ERROR: to be determined
  call co_max(c, errmsg=i)

  ! 'errmsg' argument shall be character scalar
  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_max(d, errmsg=character_array)
 
  ! the error is seen as too many arguments to the co_max() call
  !ERROR: too many actual arguments for intrinsic 'co_max'
  call co_max(r, result_image=1, stat=status, errmsg=message, 3.4)
  
  ! keyword argument with incorrect name
  !ERROR: unknown keyword argument to intrinsic 'co_max'
  call co_max(fake=3.4)
  
end program test_co_max
