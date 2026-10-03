! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic -Werror

program test
  integer :: i
  volatile :: i
  real :: volatileVar
  volatile :: volatileVar
  real :: asyncVar
  asynchronous :: asyncVar
  !WARNING: VOLATILE actual argument 'i' should be passed via an explicit interface [-Wimplicit-interface-actual]
  call sub(i)
  !WARNING: VOLATILE actual argument 'volatilevar' should be passed via an explicit interface [-Wimplicit-interface-actual]
  call implicitVol(volatileVar)
  !WARNING: ASYNCHRONOUS actual argument 'asyncvar' should be passed via an explicit interface [-Wimplicit-interface-actual]
  call implicitAsync(asyncVar)
end

subroutine sub(i)
  integer :: i
end subroutine
