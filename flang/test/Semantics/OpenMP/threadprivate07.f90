! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp

! Check Threadprivate Directive with local variable of a BLOCK construct.

program main
  call sub1()
  print *, 'pass'
end program main

subroutine sub1()
  BLOCK
    integer, save :: a
    !$omp threadprivate(a)
  END BLOCK
end subroutine
