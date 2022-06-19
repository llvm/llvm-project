! RUN: %python %S/test_errors.py %s %flang_fc1
! XFAIL: *
! This test checks for semantic errors in co_broadcast subroutine calls based on
! the co_broadcast interface defined in section 16.9.46 of the Fortran 2018 standard.
! To Do: add co_broadcast to the list of intrinsics

program test_co_broadcast
  implicit none

  type foo_t
  end type

  integer          i, integer_array(1), coindexed_integer[*], status
  character(len=1) c, character_array(1), coindexed_character[*], message
  double precision d, double_precision_array(1)
  type(foo_t)      f
  real             r, real_array(1), coindexed_real[*]
  complex          z, complex_array
  logical bool

  !___ standard-conforming calls with no keyword arguments ___
  call co_broadcast(i, 1)
  call co_broadcast(c, 1)
  call co_broadcast(d, 1)
  call co_broadcast(f, 1)
  call co_broadcast(r, 1)
  call co_broadcast(z, 1)
  call co_broadcast(i, 1, status)
  call co_broadcast(i, 1, status, message)

  !___ standard-conforming calls with keyword arguments ___

  ! all arguments present
  call co_broadcast(a=i, source_image=1, stat=status, errmsg=message) 
  call co_broadcast(source_image=1, a=i, errmsg=message, stat=status) 

  ! one optional argument not present
  call co_broadcast(a=d, source_image=1,              errmsg=message)
  call co_broadcast(a=f, source_image=1, stat=status                )

  ! two optional arguments not present
  call co_broadcast(a=r, source_image=1                             ) 

  !___ non-standard-conforming calls ___

  !ERROR: missing mandatory 'a=' argument
  call co_broadcast(source_image=1, stat=status, errmsg=message)

  !ERROR: missing mandatory 'source_image=' argument
  call co_broadcast(a=c, stat=status, errmsg=message) 

  ! argument 'a' is intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' must be definable
  call co_broadcast(a=1+1, source_image=1)
  
  ! argument 'a' shall not be a coindexed object
  !ERROR: to be determined
  call co_broadcast(a=coindexed_real[1], source_image=1)
  
  ! 'source_image' argument shall be an integer
  !ERROR: Actual argument for 'source_image=' has bad type 'LOGICAL(4)'
  call co_broadcast(i, source_image=bool)
  
  ! 'source_image' argument shall be an integer scalar
  !ERROR: 'source_image=' argument has unacceptable rank 1
  call co_broadcast(c, source_image=integer_array)
  
  ! 'stat' argument shall be intent(out)
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' must be definable
  call co_broadcast(a=i, source_image=1, stat=1+1, errmsg=message)

  ! 'stat' argument shall be noncoindexed
  !ERROR: to be determined
  call co_broadcast(d, stat=coindexed_integer[1], source_image=1)
 
  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_broadcast(r, stat=message, source_image=1)
 
  ! 'stat' argument shall be an integer scalar
  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_broadcast(i, stat=integer_array, source_image=1)
 
  ! 'errmsg' argument shall be intent(inout)
  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' must be definable
  call co_broadcast(a=i, source_image=1, stat=status, errmsg='c')

  ! 'errmsg' argument shall be noncoindexed
  !ERROR: to be determined
  call co_broadcast(c, errmsg=coindexed_character[1], source_image=1)
 
  ! 'errmsg' argument shall be character scalar
  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_broadcast(d, errmsg=character_array, source_image=1)
 
  ! the error is seen as too many arguments to the co_broadcast() call
  !ERROR: too many actual arguments for intrinsic 'co_broadcast'
  call co_broadcast(r, source_image=1, stat=status, errmsg=message, 3.4)
  
  ! keyword argument with incorrect name
  !ERROR: unknown keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(fake=3.4)
  
end program test_co_broadcast
