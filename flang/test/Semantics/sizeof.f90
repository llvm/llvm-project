! RUN: %python %S/test_errors.py %s %flang_fc1
  character(len=20) :: a, b
  if (sizeof(a) == sizeof(x=b)) then
    print *, "pass"
  else
    print *, "fail"
  end if
  !ERROR: unknown keyword argument to intrinsic 'sizeof'
  print *, sizeof(a=a)
end