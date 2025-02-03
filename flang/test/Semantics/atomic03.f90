! RUN: %python %S/test_errors.py %s %flang_fc1
! This test checks for semantic errors in atomic_cas subroutine calls based on
! the interface defined in section 16.9.22 of the Fortran 2018 standard.

program test_atomic_cas
  use iso_fortran_env, only: atomic_int_kind, atomic_logical_kind
  implicit none(external, type)

  integer(kind=atomic_int_kind) :: int_scalar_coarray[*], non_scalar_coarray(10)[*], non_coarray
  integer(kind=atomic_int_kind) :: repeated_atom[*], array(10)
  integer(kind=atomic_int_kind) :: old_int, compare_int, new_int, non_scalar_int(10)
  integer(kind=atomic_int_kind) :: repeated_old, repeated_compare, repeated_new
  integer :: status, default_kind_coarray[*], coindexed_status[*], extra_arg, repeated_status, status_array(10)
  integer(kind=1) :: kind1_coarray[*], old_kind1, compare_kind1, new_kind1
  real :: non_integer_coarray[*], non_int_or_logical
  logical(kind=atomic_logical_kind) :: logical_scalar_coarray[*], non_scalar_logical_coarray(10)[*], non_coarray_logical
  logical(kind=atomic_logical_kind) :: old_logical, compare_logical, new_logical, non_scalar_logical(10)
  logical(kind=1) :: kind1_logical_coarray[*], old_logical_kind1, compare_logical_kind1, new_logical_kind1
  logical :: non_integer, default_kind_logical_coarray[*]

  !___ standard-conforming calls ___
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int)
  call atomic_cas(int_scalar_coarray[1], old_int, compare_int, new_int)
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, status)
  call atomic_cas(int_scalar_coarray[1], old_int, compare_int, new_int, status)
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, stat=status)
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new=new_int, stat=status)
  call atomic_cas(int_scalar_coarray, old_int, compare=compare_int, new=new_int, stat=status)
  call atomic_cas(int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, stat=status)
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int)
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, stat=status)
  call atomic_cas(new=new_int, old=old_int, atom=int_scalar_coarray, stat=status, compare=compare_int)

  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new_logical)
  call atomic_cas(logical_scalar_coarray[1], old_logical, compare_logical, new_logical)
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new_logical, status)
  call atomic_cas(logical_scalar_coarray[1], old_logical, compare_logical, new_logical, status)
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new_logical, stat=status)
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new=new_logical, stat=status)
  call atomic_cas(logical_scalar_coarray, old_logical, compare=compare_logical, new=new_logical, stat=status)
  call atomic_cas(logical_scalar_coarray, old=old_logical, compare=compare_logical, new=new_logical, stat=status)
  call atomic_cas(atom=logical_scalar_coarray, old=old_logical, compare=compare_logical, new=new_logical)
  call atomic_cas(atom=logical_scalar_coarray, old=old_logical, compare=compare_logical, new=new_logical, stat=status)
  call atomic_cas(new=new_logical, old=old_logical, atom=logical_scalar_coarray, stat=status, compare=compare_logical)

  !___ non-standard-conforming calls ___

! mismatches where 'atom' is not a scalar coarray or coindexed object

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_scalar_coarray, old_int, compare_int, new_int)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_scalar_coarray[1], old_int, compare_int, new_int)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_scalar_logical_coarray, old_logical, compare_logical, new_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_scalar_logical_coarray[1], old_logical, compare_logical, new_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_coarray, old_int, compare_int, new_int)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(non_coarray_logical, old_logical, compare_logical, new_logical)

  !ERROR: 'atom=' argument must be a scalar coarray or coindexed object for intrinsic 'atomic_cas'
  call atomic_cas(array, old_int, compare_int, new_int)

! mismatches where 'atom' has wrong kind

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(4)'
  call atomic_cas(default_kind_coarray, old_int, compare_int, new_int)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_int_kind, but is 'INTEGER(1)'
  call atomic_cas(kind1_coarray, old_int, compare_int, new_int)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_logical_kind, but is 'LOGICAL(4)'
  call atomic_cas(default_kind_logical_coarray, old_logical, compare_logical, new_logical)

  !ERROR: Actual argument for 'atom=' must have kind=atomic_logical_kind, but is 'LOGICAL(1)'
  call atomic_cas(kind1_logical_coarray, old_logical, compare_logical, new_logical)

! mismatch where 'atom' has wrong type

  !ERROR: Actual argument for 'atom=' has bad type 'REAL(4)'
  call atomic_cas(non_integer_coarray, old_int, compare_int, new_int)

! mismatches where 'old', 'compare', 'new' arguments don't match type of 'atom'

  !ERROR: Actual argument for 'old=' must have same type and kind as 'atom=', but is 'LOGICAL(8)'
  call atomic_cas(int_scalar_coarray, old_logical, compare_int, new_int)

  !ERROR: Actual argument for 'compare=' must have same type and kind as 'atom=', but is 'LOGICAL(8)'
  call atomic_cas(int_scalar_coarray, old_int, compare_logical, new_int)

  !ERROR: Actual argument for 'new=' must have same type and kind as 'atom=', but is 'LOGICAL(8)'
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_logical)

  !ERROR: Actual argument for 'old=' must have same type and kind as 'atom=', but is 'INTEGER(8)'
  call atomic_cas(logical_scalar_coarray, old_int, compare_logical, new_logical)

  !ERROR: Actual argument for 'compare=' must have same type and kind as 'atom=', but is 'INTEGER(8)'
  call atomic_cas(logical_scalar_coarray, old_logical, compare_int, new_logical)

  !ERROR: Actual argument for 'new=' must have same type and kind as 'atom=', but is 'INTEGER(8)'
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new_int)

! mismatches where 'old', 'compare', 'new' arguments matches type of 'atom' but doesn't match kind of 'atom'

  !ERROR: Actual argument for 'old=' must have same type and kind as 'atom=', but is 'INTEGER(1)'
  call atomic_cas(int_scalar_coarray, old_kind1, compare_int, new_int)

  !ERROR: Actual argument for 'compare=' must have same type and kind as 'atom=', but is 'INTEGER(1)'
  call atomic_cas(int_scalar_coarray, old_int, compare_kind1, new_int)

  !ERROR: Actual argument for 'new=' must have same type and kind as 'atom=', but is 'INTEGER(1)'
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_kind1)

  !ERROR: Actual argument for 'old=' must have same type and kind as 'atom=', but is 'LOGICAL(1)'
  call atomic_cas(logical_scalar_coarray, old_logical_kind1, compare_logical, new_logical)

  !ERROR: Actual argument for 'compare=' must have same type and kind as 'atom=', but is 'LOGICAL(1)'
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical_kind1, new_logical)

  !ERROR: Actual argument for 'new=' must have same type and kind as 'atom=', but is 'LOGICAL(1)'
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, new_logical_kind1)

! mismatches where 'old', 'compare', 'new' arguments have incorrect rank

  !ERROR: 'old=' argument has unacceptable rank 1
  call atomic_cas(int_scalar_coarray, non_scalar_int, compare_int, new_int)

  !ERROR: 'compare=' argument has unacceptable rank 1
  call atomic_cas(int_scalar_coarray, old_int, non_scalar_int, new_int)

  !ERROR: 'new=' argument has unacceptable rank 1
  call atomic_cas(int_scalar_coarray, old_int, compare_int, non_scalar_int)

  !ERROR: 'old=' argument has unacceptable rank 1
  call atomic_cas(logical_scalar_coarray, non_scalar_logical, compare_logical, new_logical)

  !ERROR: 'compare=' argument has unacceptable rank 1
  call atomic_cas(logical_scalar_coarray, old_logical, non_scalar_logical, new_logical)

  !ERROR: 'new=' argument has unacceptable rank 1
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, non_scalar_logical)

! mismatches where 'old', 'compare', 'new' arguments have incorrect type

  !ERROR: Actual argument for 'old=' has bad type 'REAL(4)'
  call atomic_cas(int_scalar_coarray, non_int_or_logical, compare_int, new_int)

  !ERROR: Actual argument for 'compare=' has bad type 'REAL(4)'
  call atomic_cas(int_scalar_coarray, old_int, non_int_or_logical, new_int)

  !ERROR: Actual argument for 'new=' has bad type 'REAL(4)'
  call atomic_cas(int_scalar_coarray, old_int, compare_int, non_int_or_logical)

  !ERROR: Actual argument for 'old=' has bad type 'REAL(4)'
  call atomic_cas(logical_scalar_coarray, non_int_or_logical, compare_logical, new_logical)

  !ERROR: Actual argument for 'compare=' has bad type 'REAL(4)'
  call atomic_cas(logical_scalar_coarray, old_logical, non_int_or_logical, new_logical)

  !ERROR: Actual argument for 'new=' has bad type 'REAL(4)'
  call atomic_cas(logical_scalar_coarray, old_logical, compare_logical, non_int_or_logical)

! mismatches on 'stat' argument

  !ERROR: Actual argument for 'stat=' has bad type 'LOGICAL(4)'
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, non_integer)

  !ERROR: 'stat=' argument has unacceptable rank 1
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, status_array)

  !ERROR: 'stat' argument to 'atomic_cas' may not be a coindexed object
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, coindexed_status[1])

  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'stat=' is not definable
  !BECAUSE: '1_4' is not a variable or pointer
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, 1)

! missing mandatory arguments

  !ERROR: missing mandatory 'atom=' argument
  call atomic_cas()

  !ERROR: missing mandatory 'atom=' argument
  call atomic_cas(old=old_int, compare=compare_int, new=new_int, stat=status)

  !ERROR: missing mandatory 'old=' argument
  call atomic_cas(atom=int_scalar_coarray, compare=compare_int, new=new_int, stat=status)

  !ERROR: missing mandatory 'compare=' argument
  call atomic_cas(atom=int_scalar_coarray, old=old_int, new=new_int, stat=status)

  !ERROR: missing mandatory 'new=' argument
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, stat=status)

  !ERROR: missing mandatory 'new=' argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int)

! too many arguments or repeated arguments

  !ERROR: too many actual arguments for intrinsic 'atomic_cas'
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, status, extra_arg)

  !ERROR: repeated keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, atom=repeated_atom, old=old_int, compare=compare_int, new=new_int)

  !ERROR: repeated keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, old=repeated_old, compare=compare_int, new=new_int)

  !ERROR: repeated keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, compare=repeated_compare, new=new_int)

  !ERROR: repeated keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, new=repeated_new)

  !ERROR: repeated keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, stat=status, stat=repeated_status)

  !ERROR: keyword argument to intrinsic 'atomic_cas' was supplied positionally by an earlier actual argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, atom=repeated_atom)

  !ERROR: keyword argument to intrinsic 'atomic_cas' was supplied positionally by an earlier actual argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, old=repeated_old)

  !ERROR: keyword argument to intrinsic 'atomic_cas' was supplied positionally by an earlier actual argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, compare=repeated_compare)

  !ERROR: keyword argument to intrinsic 'atomic_cas' was supplied positionally by an earlier actual argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, new=repeated_new)

  !ERROR: keyword argument to intrinsic 'atomic_cas' was supplied positionally by an earlier actual argument
  call atomic_cas(int_scalar_coarray, old_int, compare_int, new_int, status, stat=repeated_status)

! typo in keyword arguments

  !ERROR: unknown keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atomic=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, ol=old_int, compare=compare_int, new=new_int, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, comp=compare_int, new=new_int, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, nw=new_int, stat=status)

  !ERROR: unknown keyword argument to intrinsic 'atomic_cas'
  call atomic_cas(atom=int_scalar_coarray, old=old_int, compare=compare_int, new=new_int, status=status)

end program test_atomic_cas
