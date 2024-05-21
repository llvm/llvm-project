! RUN: not %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s 2>&1 | FileCheck %s
! OpenMP Version 4.5
! 2.7.1 Loop Construct
program omp_doCollapse
  integer:: i
  !$omp parallel do collapse(2)
    do i = 1, 3
      !CHECK: Loop control is not present in the DO LOOP
      !CHECK: associated with the enclosing LOOP construct
      do
      end do
    end do
end program omp_doCollapse

