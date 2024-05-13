! RUN: %python %S/test_errors.py %s %flang_fc1
! NULL() intrinsic function error tests
program test_random_seed
  integer :: size_arg
  integer, parameter :: size_arg_const = 343
  integer, dimension(3), parameter :: put_arg = [9,8,7]
  integer  :: get_arg_scalar
  integer, dimension(3) :: get_arg
  integer, dimension(3),parameter :: get_arg_const = [8,7,6]
  call random_seed()
  call random_seed(size_arg)
  call random_seed(size=size_arg)
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'size=' is not definable
  !BECAUSE: '343_4' is not a variable or pointer
  call random_seed(size_arg_const) ! error, size arg must be definable
  !ERROR: 'size=' argument has unacceptable rank 1
  call random_seed([1, 2, 3, 4]) ! Error, must be a scalar
  call random_seed(put = [1, 2, 3, 4])
  call random_seed(put = put_arg)
  !ERROR: 'size=' argument has unacceptable rank 1
  call random_seed(get_arg) ! Error, must be a scalar
  call random_seed(get=get_arg)
  !ERROR: 'get=' argument has unacceptable rank 0
  call random_seed(get=get_arg_scalar) ! Error, GET arg must be of rank 1
  !ERROR: Actual argument associated with INTENT(OUT) dummy argument 'get=' is not definable
  !BECAUSE: '[INTEGER(4)::8_4,7_4,6_4]' is not a variable or pointer
  call random_seed(get=get_arg_const) ! Error, GET arg must be definable
  !ERROR: RANDOM_SEED must have either 1 or no arguments
  call random_seed(size_arg, get_arg) ! Error, only 0 or 1 argument
end program
