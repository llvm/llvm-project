!RUN: %python %S/test_errors.py %s %flang_fc1
subroutine s(i)
  type(*) :: i
  !ERROR: TYPE(*) dummy argument may only be used as an actual argument
  !ERROR: Assumed-type entity 'i' must be a dummy argument
  !ERROR: Must have INTEGER type, but is TYPE(*)
  print *, [(i, i = 1,1)]
end
