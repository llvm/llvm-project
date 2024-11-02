! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in co_broadcast subroutine calls based on
! the co_broadcast interface defined in section 16.9.46 of the Fortran 2018 standard.
! To Do: add co_broadcast to the list of intrinsics

program test_co_broadcast
  implicit none

  type foo_t
  end type

  integer          i, integer_array(1), coindexed_integer[*], status, coindexed_source_image[*], repeated_status
  character(len=1) c, character_array(1), coindexed_character[*], message, repeated_message
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
  call co_broadcast(a=r, source_image=coindexed_source_image        )

  !___ non-standard-conforming calls ___

  !ERROR: missing mandatory 'a=' argument
  call co_broadcast()

  !ERROR: repeated keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(a=i, a=c)

  !ERROR: repeated keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(d, source_image=1, source_image=3)

  !ERROR: repeated keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(d, 1, stat=status, stat=repeated_status)

  !ERROR: repeated keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(d, 1, status, errmsg=message, errmsg=repeated_message)

  !ERROR: keyword argument to intrinsic 'co_broadcast' was supplied positionally by an earlier actual argument
  call co_broadcast(i, 1, a=c)

  !ERROR: keyword argument to intrinsic 'co_broadcast' was supplied positionally by an earlier actual argument
  call co_broadcast(i, 1, status, source_image=1)

  !ERROR: keyword argument to intrinsic 'co_broadcast' was supplied positionally by an earlier actual argument
  call co_broadcast(i, 1, status, stat=repeated_status)

  !ERROR: keyword argument to intrinsic 'co_broadcast' was supplied positionally by an earlier actual argument
  call co_broadcast(i, 1, status, message, errmsg=repeated_message)

  !ERROR: missing mandatory 'a=' argument
  call co_broadcast(source_image=1, stat=status, errmsg=message)

  !ERROR: missing mandatory 'source_image=' argument
  call co_broadcast(c)

  !ERROR: missing mandatory 'source_image=' argument
  call co_broadcast(a=c, stat=status, errmsg=message)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
  !BECAUSE: '2_4' is not a variable or pointer
  call co_broadcast(a=1+1, source_image=1)

  !ERROR: 'a' argument to 'co_broadcast' may not be a coindexed object
  call co_broadcast(a=coindexed_real[1], source_image=1)

  ! 'source_image' argument shall be an integer
  !ERROR: Actual argument for 'source_image=' has bad type 'LOGICAL(4)'
  call co_broadcast(i, source_image=bool)

  ! 'source_image' argument shall be an integer scalar
  !ERROR: 'source_image=' argument has unacceptable rank 1
  call co_broadcast(c, source_image=integer_array)

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  !BECAUSE: '2_4' is not a variable or pointer
  call co_broadcast(a=i, source_image=1, stat=1+1, errmsg=message)

  !ERROR: 'stat' argument to 'co_broadcast' may not be a coindexed object
  call co_broadcast(d, stat=coindexed_integer[1], source_image=1)

  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_broadcast(r, stat=message, source_image=1)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_broadcast(i, stat=integer_array, source_image=1)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' is not definable
  !BECAUSE: '"c"' is not a variable or pointer
  call co_broadcast(a=i, source_image=1, stat=status, errmsg='c')

  !ERROR: 'errmsg' argument to 'co_broadcast' may not be a coindexed object
  call co_broadcast(c, errmsg=coindexed_character[1], source_image=1)

  ! 'errmsg' argument shall be a character
  !ERROR: Actual argument for 'errmsg=' has bad type 'INTEGER(4)'
  call co_broadcast(c, 1, status, i)

  ! 'errmsg' argument shall be a character
  !ERROR: Actual argument for 'errmsg=' has bad type 'INTEGER(4)'
  call co_broadcast(c, errmsg=i, source_image=1)

  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_broadcast(d, errmsg=character_array, source_image=1)

  !ERROR: too many actual arguments for intrinsic 'co_broadcast'
  call co_broadcast(r, source_image=1, stat=status, errmsg=message, 3.4)

  !ERROR: unknown keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(fake=3.4)

  !ERROR: unknown keyword argument to intrinsic 'co_broadcast'
  call co_broadcast(a=i, result_image=1, stat=status, errmsg=message)

  !ERROR: 'a' argument to 'co_broadcast' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'co_broadcast' may not be a coindexed object
  !ERROR: 'stat' argument to 'co_broadcast' may not be a coindexed object
  call co_broadcast(source_image=coindexed_source_image[1], a=coindexed_real[1], errmsg=coindexed_character[1], stat=coindexed_integer[1])

end program test_co_broadcast
