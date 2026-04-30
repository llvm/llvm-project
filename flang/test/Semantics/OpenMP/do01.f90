! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a firstprivate directive.

subroutine f00
  integer i, j, k

  !ERROR: Loop iteration variable with a predetermined data sharing attribute cannot appear in a FIRSTPRIVATE clause
  !$omp do firstprivate(k,i)
  !BECAUSE: 'i' is an iteration variable of an affected loop
  do i = 1, 10
    do j = 1, 10
      print *, "Hello"
    end do
  end do
  !$omp end do
end

! Check multiple DSA clauses
subroutine f01
  integer :: i
  !ERROR: Loop iteration variable with a predetermined data sharing attribute cannot appear in a FIRSTPRIVATE clause
  !ERROR: 'i' appears in more than one data-sharing clause on the same OpenMP directive
  !ERROR: Loop iteration variable with a predetermined data sharing attribute cannot appear in a LINEAR clause
  !$omp do firstprivate(i) lastprivate(i) linear(i)
  block
    !BECAUSE: 'i' is an iteration variable of an affected loop
    !BECAUSE: 'i' is an iteration variable of an affected loop
    do i = 1, 10
    end do
  end block
end
