! RUN: %python %S/test_errors.py %s %flang_fc1
! Confirm enforcement of F'2023 7.8 p5
subroutine subr(s,n)
  character*(*) s
  !ERROR: Array constructor implied DO loop has no iterations and indeterminate character length
  print *, [(s(1:n),j=1,0)]
  !ERROR: Array constructor implied DO loop has no iterations and indeterminate character length
  print *, [(s(1:n),j=0,1,-1)]
  !ERROR: Array constructor implied DO loop has no iterations and indeterminate character length
  print *, [(s(1:j),j=1,0)]
  print *, [(s(1:1),j=1,0)] ! ok
  print *, [character(2)::(s(1:n),j=1,0)] ! ok
  print *, [character(n)::(s(1:n),j=1,0)] ! ok
end
