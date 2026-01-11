! RUN: %python %S/test_errors.py %s %flang_fc1

program main
  implicit none
  integer :: ja, jb, j3, j4, j5
  pointer (j1,ja)
  pointer (j2,jb)
  !ERROR: Values in array constructor must have the same declared type when no explicit type appears
  if (any((/j1,j2,j3,j4,j5/)/=(/1,2,3,4,5/))) print *,'fail'
end program main
