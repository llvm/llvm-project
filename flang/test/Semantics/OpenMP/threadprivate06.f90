! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.1
! Check OpenMP construct validity for the following directives:
! 2.21.2 Threadprivate Directive

program main
  call sub1()
  print *, 'pass'
end program main

subroutine sub1()
  common /c/ a
  !$omp threadprivate(/c/)
  integer :: a

  a = 100
  call sub2()
  if (a .ne. 101) print *, 'err'

contains
  subroutine sub2()
    common /c/ a
    !$omp threadprivate(/c/)
    integer :: a

    !$omp parallel copyin(/c/)
      a = a + 1
    !$omp end parallel
  end subroutine
end subroutine
