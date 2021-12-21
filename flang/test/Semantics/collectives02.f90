! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in co_min subroutine calls based on
! the co_min interface defined in section 16.9.48 of the Fortran 2018 standard.
! To Do: add co_min to the list of intrinsics

program test_co_min
  implicit none

  integer          i, integer_array(1), coindexed_integer[*], status
  character(len=1) c, character_array(1), coindexed_character[*], message
  double precision d, double_precision_array(1)
  real             r, real_array(1), coindexed_real[*]
  complex          complex_type
  logical          bool
  
  !___ standard-conforming calls with no keyword arguments ___
  call co_min(i)
  call co_min(c)
  call co_min(d)
  call co_min(r)
  call co_min(i, 1)
  call co_min(c, 1, status)
  call co_min(d, 1, status, message)
  call co_min(r, 1, status, message)
  call co_min(integer_array)
  call co_min(character_array, 1)
  call co_min(double_precision_array, 1, status)
  call co_min(real_array, 1, status, message)

  !___ standard-conforming calls with keyword arguments ___

  ! all arguments present
  call co_min(a=i, result_image=1, stat=status, errmsg=message) 
  call co_min(result_image=1, a=i, errmsg=message, stat=status) 

  ! one optional argument not present
  call co_min(a=i,                 stat=status, errmsg=message) 
  call co_min(a=i, result_image=1,              errmsg=message)
  call co_min(a=i, result_image=1, stat=status                )

  ! two optional arguments not present
  call co_min(a=i, result_image=1                             ) 
  call co_min(a=i,                 stat=status                )
  call co_min(a=i,                              errmsg=message) 

  ! no optional arguments present
  call co_min(a=i) 

  !___ non-standard-conforming calls ___

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
  call co_min(bool)
  
  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'COMPLEX(4)'
  call co_min(complex_type)
  
  ! argument 'a' is intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' must be definable
  call co_min(a=1+1)
  
  ! argument 'a' shall not be a coindexed object
  !ERROR: to be determined
  call co_min(a=coindexed_real[1])
  
  ! 'result_image' argument shall be a scalar
  !ERROR: too many actual arguments for intrinsic 'co_min'
  call co_min(i, result_image=bool)
  
  ! 'result_image' argument shall be an integer scalar
  !ERROR: too many actual arguments for intrinsic 'co_min'
  call co_min(c, result_image=integer_array)
  
  ! 'stat' argument shall be noncoindexed
  !ERROR: to be determined
  call co_min(d, stat=coindexed_integer[1])
 
  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_min(r, stat=message)
 
  ! 'stat' argument shall be an integer scalar
  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_min(i, stat=integer_array)
 
  ! 'errmsg' argument shall be noncoindexed
  !ERROR: to be determined
  call co_min(c, errmsg=coindexed_character[1])
 
  ! 'errmsg' argument shall be character scalar
  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_min(d, errmsg=character_array)
 
  ! the error is seen as too many arguments to the co_min() call
  !ERROR: too many actual arguments for intrinsic 'co_min'
  call co_min(r, result_image=1, stat=status, errmsg=message, 3.4)
  
  ! keyword argument with incorrect name
  !ERROR: unknown keyword argument to intrinsic 'co_min'
  call co_min(fake=3.4)
  
end program test_co_min
