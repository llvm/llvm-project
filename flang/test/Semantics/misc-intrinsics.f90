! RUN: %python %S/test_errors.py %s %flang_fc1
! Miscellaneous constraint and requirement checking on intrinsics
program test_size
  real :: scalar
  real, dimension(5, 5) :: array
  call test(array)
 contains
  subroutine test(arg)
    real, dimension(5, *) :: arg
    !ERROR: A dim= argument is required for 'size' when the array is assumed-size
    print *, size(arg)
    !ERROR: missing mandatory 'dim=' argument
    print *, ubound(arg)
    !ERROR: The 'source=' argument to the intrinsic function 'shape' may not be assumed-size
    print *, shape(arg)
    !ERROR: The 'harvest=' argument to the intrinsic procedure 'random_number' may not be assumed-size
    call random_number(arg)
    !ERROR: missing mandatory 'dim=' argument
    print *, lbound(scalar)
    !ERROR: 'array=' argument has unacceptable rank 0
    print *, size(scalar)
    !ERROR: missing mandatory 'dim=' argument
    print *, ubound(scalar)
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
  end subroutine
end
