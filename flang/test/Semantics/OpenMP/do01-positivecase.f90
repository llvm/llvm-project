! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct
! The loop iteration variable may not appear in a firstprivate directive.
! A positive case

!DEF: /OMP_DO MainProgram
program OMP_DO
  !DEF: /OMP_DO/i ObjectEntity INTEGER(4)
  integer i

  !$omp do  firstprivate(k)
  !DEF: /OMP_DO/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    print *, "Hello"
  end do
  !$omp end do

end program OMP_DO
