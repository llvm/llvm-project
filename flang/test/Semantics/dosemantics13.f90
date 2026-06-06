! RUN: %python %S/test_errors.py %s %flang_fc1
program main

  integer j, k

  lab1: do j=1,10
    cycle lab1
    exit lab1
  end do lab1

  lab2: do 2 j=1,10
    cycle lab2
    exit lab2
  2 end do lab2

  lab3: do 3 j=1,10
    cycle lab3
    exit lab3
    !ERROR: DO construct name required but missing
  3 end do

  do 4 j=1,10
  !ERROR: Unexpected DO construct name 'lab4'
  4 end do lab4

  lab5: do 5 j=1,10
  !ERROR: END DO statement must have the label '5' matching its DO statement
  666 end do lab5
end
