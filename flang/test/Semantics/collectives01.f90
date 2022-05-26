! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in co_sum subroutine calls based on
! the co_reduce interface defined in section 16.9.50 of the Fortran 2018 standard.

program test_co_sum
  implicit none

  integer i, status, integer_array(1), coindexed_integer[*], coindexed_result_image[*]
  complex c, complex_array(1,1,1, 1,1,1, 1,1,1, 1,1,1, 1,1,1) 
  double precision d, double_precision_array(1)
  real r, real_array(1), coindexed_real[*]

  character(len=1) message, coindexed_character[*], character_array(1)
  logical bool
  
  !___ standard-conforming calls with no keyword arguments ___
  call co_sum(i)
  call co_sum(c)
  call co_sum(d)
  call co_sum(r)
  call co_sum(i, 1)
  call co_sum(c, 1, status)
  call co_sum(d, 1, status, message)
  call co_sum(r, 1, status, message)
  call co_sum(integer_array)
  call co_sum(complex_array, 1)
  call co_sum(double_precision_array, 1, status)
  call co_sum(real_array, 1, status, message)

  !___ standard-conforming calls with keyword arguments ___

  ! all arguments present
  call co_sum(a=i, result_image=1, stat=status, errmsg=message) 
  call co_sum(result_image=1, a=i, errmsg=message, stat=status)

  ! one optional argument not present
  call co_sum(a=i,                 stat=status, errmsg=message) 
  call co_sum(a=i, result_image=1,              errmsg=message)
  call co_sum(a=i, result_image=1, stat=status                )

  ! two optional arguments not present
  call co_sum(a=i, result_image=1                             ) 
  call co_sum(a=i,                 stat=status                )
  call co_sum(a=i,                              errmsg=message) 
  call co_sum(a=i, result_image=coindexed_result_image[1])

  ! no optional arguments present
  call co_sum(a=i                                             ) 

  !___ non-standard-conforming calls ___

  !ERROR: missing mandatory 'a=' argument
  call co_sum()

  !ERROR: missing mandatory 'a=' argument
  call co_sum(result_image=1, stat=status, errmsg=message)

  !ERROR: repeated keyword argument to intrinsic 'co_sum'
  call co_sum(a=i, a=c)

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
  call co_sum(bool)
  
  ! argument 'a' is intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' must be definable
  call co_sum(a=1+1)
  
  !ERROR: 'a' argument to 'co_sum' may not be a coindexed object
  call co_sum(a=coindexed_real[1])
  
  ! 'result_image' argument shall be a integer
  !ERROR: Actual argument for 'result_image=' has bad type 'LOGICAL(4)'
  call co_sum(i, result_image=bool)
  
  ! 'result_image' argument shall be an integer scalar
  !ERROR: 'result_image=' argument has unacceptable rank 1
  call co_sum(c, result_image=integer_array)

  ! argument 'stat' shall be intent(out)
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' must be definable
  call co_sum(a=i, result_image=1, stat=1+1, errmsg=message)

  !ERROR: 'stat' argument to 'co_sum' may not be a coindexed object
  call co_sum(d, stat=coindexed_integer[1])

  !ERROR: 'stat' argument to 'co_sum' may not be a coindexed object
  call co_sum(stat=coindexed_integer[1], a=d)
 
  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_sum(r, stat=message)
 
  ! 'stat' argument shall be an integer scalar
  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_sum(i, stat=integer_array)
 
  ! 'errmsg' argument shall be intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' must be definable
  call co_sum(a=i, result_image=1, stat=status, errmsg='c')
  
  !ERROR: 'errmsg' argument to 'co_sum' may not be a coindexed object
  call co_sum(c, errmsg=coindexed_character[1])

  ! 'errmsg' argument shall be a character
  !ERROR: Actual argument for 'errmsg=' has bad type 'INTEGER(4)'
  call co_sum(c, errmsg=i)
 
  ! 'errmsg' argument shall be character scalar
  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_sum(d, errmsg=character_array)
 
  !ERROR: too many actual arguments for intrinsic 'co_sum'
  call co_sum(r, result_image=1, stat=status, errmsg=message, 3.4)
  
  ! keyword argument with incorrect name
  !ERROR: unknown keyword argument to intrinsic 'co_sum'
  call co_sum(fake=3.4)

  !ERROR: 'a' argument to 'co_sum' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'co_sum' may not be a coindexed object
  !ERROR: 'stat' argument to 'co_sum' may not be a coindexed object
  call co_sum(result_image=coindexed_result_image[1], a=coindexed_real[1], errmsg=coindexed_character[1], stat=coindexed_integer[1])
  
end program test_co_sum
