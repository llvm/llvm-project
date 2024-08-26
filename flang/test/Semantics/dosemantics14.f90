! RUN: %python %S/test_errors.py %s %flang_fc1
subroutine bad1
  lab1: do 1 j=1,10
  1 continue
!ERROR: expected 'END DO'
end

subroutine bad2
  lab2: do 2 j=1,10
  2 k = j
!ERROR: expected 'END DO'
end
