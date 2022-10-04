! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in co_min subroutine calls based on
! the co_min interface defined in section 16.9.48 of the Fortran 2018 standard.

program test_co_min
  implicit none

  integer          i, integer_array(1), coindexed_integer[*], status, coindexed_result_image[*], repeated_status
  character(len=1) c, character_array(1), coindexed_character[*], message, repeated_message
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
  call co_min(a=i, result_image=coindexed_result_image[1]     )

  ! no optional arguments present
  call co_min(a=i)

  !___ non-standard-conforming calls ___

  !ERROR: missing mandatory 'a=' argument
  call co_min()

  !ERROR: repeated keyword argument to intrinsic 'co_min'
  call co_min(a=i, a=c)

  !ERROR: repeated keyword argument to intrinsic 'co_min'
  call co_min(d, result_image=1, result_image=3)

  !ERROR: repeated keyword argument to intrinsic 'co_min'
  call co_min(d, 1, stat=status, stat=repeated_status)

  !ERROR: repeated keyword argument to intrinsic 'co_min'
  call co_min(d, 1, status, errmsg=message, errmsg=repeated_message)

  !ERROR: keyword argument to intrinsic 'co_min' was supplied positionally by an earlier actual argument
  call co_min(i, 1, a=c)

  !ERROR: keyword argument to intrinsic 'co_min' was supplied positionally by an earlier actual argument
  call co_min(i, 1, status, result_image=1)

  !ERROR: keyword argument to intrinsic 'co_min' was supplied positionally by an earlier actual argument
  call co_min(i, 1, status, stat=repeated_status)

  !ERROR: keyword argument to intrinsic 'co_min' was supplied positionally by an earlier actual argument
  call co_min(i, 1, status, message, errmsg=repeated_message)

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
  call co_min(bool)

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'COMPLEX(4)'
  call co_min(complex_type)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' must be definable
  call co_min(a=1+1)

  !ERROR: 'a' argument to 'co_min' may not be a coindexed object
  call co_min(a=coindexed_real[1])

  !ERROR: Actual argument for 'result_image=' has bad type 'LOGICAL(4)'
  call co_min(i, result_image=bool)

  !ERROR: 'result_image=' argument has unacceptable rank 1
  call co_min(c, result_image=integer_array)

  !ERROR: 'stat' argument to 'co_min' may not be a coindexed object
  call co_min(d, stat=coindexed_integer[1])

  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_min(r, stat=message)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_min(i, stat=integer_array)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' must be definable
  call co_min(a=i, result_image=1, stat=status, errmsg='c')

  !ERROR: 'errmsg' argument to 'co_min' may not be a coindexed object
  call co_min(c, errmsg=coindexed_character[1])

  ! 'errmsg' argument shall be a character
  !ERROR: Actual argument for 'errmsg=' has bad type 'INTEGER(4)'
  call co_min(c, errmsg=i)

  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_min(d, errmsg=character_array)

  !ERROR: too many actual arguments for intrinsic 'co_min'
  call co_min(r, result_image=1, stat=status, errmsg=message, 3.4)

  !ERROR: unknown keyword argument to intrinsic 'co_min'
  call co_min(fake=3.4)

  !ERROR: 'a' argument to 'co_min' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'co_min' may not be a coindexed object
  !ERROR: 'stat' argument to 'co_min' may not be a coindexed object
  call co_min(result_image=coindexed_result_image[1], a=coindexed_real[1], errmsg=coindexed_character[1], stat=coindexed_integer[1])

end program test_co_min
