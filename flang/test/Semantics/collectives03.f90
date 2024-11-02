! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in co_max subroutine calls based on
! the co_max interface defined in section 16.9.47 of the Fortran 2018 standard.

program test_co_max
  implicit none

  integer          i, integer_array(1), coindexed_integer[*], status, coindexed_result_image[*], repeated_status
  character(len=1) c, character_array(1), coindexed_character[*], message, repeated_message
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
  call co_max(a=i, result_image=coindexed_result_image[1]     )

  ! no optional arguments present
  call co_max(a=i)

  !___ non-standard-conforming calls ___

  !ERROR: missing mandatory 'a=' argument
  call co_max()

  !ERROR: repeated keyword argument to intrinsic 'co_max'
  call co_max(a=i, a=c)

  !ERROR: repeated keyword argument to intrinsic 'co_max'
  call co_max(d, result_image=1, result_image=3)

  !ERROR: repeated keyword argument to intrinsic 'co_max'
  call co_max(d, 1, stat=status, stat=repeated_status)

  !ERROR: repeated keyword argument to intrinsic 'co_max'
  call co_max(d, 1, status, errmsg=message, errmsg=repeated_message)

  !ERROR: keyword argument to intrinsic 'co_max' was supplied positionally by an earlier actual argument
  call co_max(i, 1, a=c)

  !ERROR: keyword argument to intrinsic 'co_max' was supplied positionally by an earlier actual argument
  call co_max(i, 1, status, result_image=1)

  !ERROR: keyword argument to intrinsic 'co_max' was supplied positionally by an earlier actual argument
  call co_max(i, 1, status, stat=repeated_status)

  !ERROR: keyword argument to intrinsic 'co_max' was supplied positionally by an earlier actual argument
  call co_max(i, 1, status, message, errmsg=repeated_message)

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'LOGICAL(4)'
  call co_max(bool)

  ! argument 'a' shall be of numeric type
  !ERROR: Actual argument for 'a=' has bad type 'COMPLEX(4)'
  call co_max(complex_type)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'a=' is not definable
  !BECAUSE: '2_4' is not a variable or pointer
  call co_max(a=1+1)

  !ERROR: 'a' argument to 'co_max' may not be a coindexed object
  call co_max(a=coindexed_real[1])

  !ERROR: Actual argument for 'result_image=' has bad type 'LOGICAL(4)'
  call co_max(i, result_image=bool)

  !ERROR: 'result_image=' argument has unacceptable rank 1
  call co_max(c, result_image=integer_array)

  !ERROR: 'stat' argument to 'co_max' may not be a coindexed object
  call co_max(d, stat=coindexed_integer[1])

  ! 'stat' argument shall be an integer
  !ERROR: Actual argument for 'stat=' has bad type 'CHARACTER(KIND=1,LEN=1_8)'
  call co_max(r, stat=message)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call co_max(i, stat=integer_array)

  !ERROR: Actual argument associated with INTENT(IN OUT) dummy argument 'errmsg=' is not definable
  !BECAUSE: '"c"' is not a variable or pointer
  call co_max(a=i, result_image=1, stat=status, errmsg='c')

  !ERROR: 'errmsg' argument to 'co_max' may not be a coindexed object
  call co_max(c, errmsg=coindexed_character[1])

  ! 'errmsg' argument shall be a character
  !ERROR: Actual argument for 'errmsg=' has bad type 'INTEGER(4)'
  call co_max(c, errmsg=i)

  !ERROR: 'errmsg=' argument has unacceptable rank 1
  call co_max(d, errmsg=character_array)

  !ERROR: too many actual arguments for intrinsic 'co_max'
  call co_max(r, result_image=1, stat=status, errmsg=message, 3.4)

  !ERROR: unknown keyword argument to intrinsic 'co_max'
  call co_max(fake=3.4)

  !ERROR: 'a' argument to 'co_max' may not be a coindexed object
  !ERROR: 'errmsg' argument to 'co_max' may not be a coindexed object
  !ERROR: 'stat' argument to 'co_max' may not be a coindexed object
  call co_max(result_image=coindexed_result_image[1], a=coindexed_real[1], errmsg=coindexed_character[1], stat=coindexed_integer[1])

end program test_co_max
