! RUN: %python %S/test_errors.py %s %flang_fc1
! Miscellaneous constraint and requirement checking on intrinsics
program test_size
  real :: scalar
  real, dimension(5, 5) :: array
  call test(array, array)
 contains
  subroutine test(arg, assumedRank)
    real, dimension(5, *) :: arg
    real, dimension(..) :: assumedRank
    !ERROR: A dim= argument is required for 'size' when the array is assumed-size
    print *, size(arg)
    !ERROR: A dim= argument is required for 'ubound' when the array is assumed-size
    print *, ubound(arg)
    !ERROR: The 'source=' argument to the intrinsic function 'shape' may not be assumed-size
    print *, shape(arg)
    !ERROR: The 'harvest=' argument to the intrinsic procedure 'random_number' may not be assumed-size
    call random_number(arg)
    !ERROR: 'array=' argument has unacceptable rank 0
    print *, lbound(scalar)
    !ERROR: 'array=' argument has unacceptable rank 0
    print *, size(scalar)
    !ERROR: 'array=' argument has unacceptable rank 0
    print *, ubound(scalar)
    !ERROR: DIM=0 dimension must be positive
    print *, lbound(arg, 0)
    !ERROR: DIM=0 dimension must be positive
    print *, lbound(assumedRank, 0)
    !ERROR: DIM=666 dimension is too large for any array (maximum rank 15)
    print *, lbound(assumedRank, 666)
    !ERROR: DIM=0 dimension must be positive
    print *, ubound(arg, 0)
    !ERROR: DIM=2 dimension is out of range for rank-2 assumed-size array
    print *, ubound(arg, 2)
    !ERROR: DIM=0 dimension must be positive
    print *, ubound(assumedRank, 0)
    !ERROR: DIM=666 dimension is too large for any array (maximum rank 15)
    print *, ubound(assumedRank, 666)
    select rank(assumedRank)
    rank(1)
      !ERROR: DIM=2 dimension is out of range for rank-1 array
      print *, lbound(assumedRank, dim=2)
      !ERROR: DIM=2 dimension is out of range for rank-1 array
      print *, ubound(assumedRank, dim=2)
    rank(*)
      !ERROR: A dim= argument is required for 'size' when the array is assumed-size
      print *, size(assumedRank)
      !ERROR: A dim= argument is required for 'ubound' when the array is assumed-size
      print *, ubound(assumedRank)
      !ERROR: The 'source=' argument to the intrinsic function 'shape' may not be assumed-size
      print *, shape(assumedRank)
      !ERROR: The 'harvest=' argument to the intrinsic procedure 'random_number' may not be assumed-size
      call random_number(assumedRank)
      !ERROR: DIM=0 dimension must be positive
      print *, lbound(assumedRank, 0)
      !ERROR: DIM=0 dimension must be positive
      print *, ubound(assumedRank, 0)
      !ERROR: DIM=1 dimension is out of range for rank-1 assumed-size array
      print *, ubound(assumedRank, 1)
      !ERROR: DIM=2 dimension is out of range for rank-1 array
      print *, lbound(assumedRank, dim=2)
      !ERROR: DIM=2 dimension is out of range for rank-1 array
      print *, ubound(assumedRank, dim=2)
    end select
    ! But these cases are fine:
    print *, size(arg, dim=1)
    print *, ubound(arg, dim=1)
    print *, lbound(arg)
    print *, size(array)
    print *, ubound(array)
    print *, lbound(array)
    print *, size(arg(:,1))
    print *, ubound(arg(:,1))
    print *, shape(scalar)
    print *, shape(arg(:,1))
    print *, lbound(assumedRank, dim=2) ! can't check until run time
    print *, ubound(assumedRank, dim=2)
    select rank(assumedRank)
    rank(3)
      print *, lbound(assumedRank, dim=2)
      print *, ubound(assumedRank, dim=2)
    rank(*)
      print *, lbound(assumedRank, dim=1)
    rank default
      print *, lbound(assumedRank, dim=2)
      print *, ubound(assumedRank, dim=2)
    end select
  end subroutine
end
