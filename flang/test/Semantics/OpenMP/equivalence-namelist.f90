! RUN: %python %S/../test_errors.py %s %flang -fopenmp

! The openmp standard only dissallows namelist for privatization, but flang
! also does not allow it for reduction as this would be difficult to support.
!
! Variables in equivalence with variables in the namelist pose the same
! implementation problems.

subroutine test01()
  integer::va
  equivalence (va,vva)
  namelist /na1/vva
  va=1

!ERROR: Variable 'va' in NAMELIST cannot be in a REDUCTION clause
!$omp parallel reduction(+:va)
  write(*,na1)
!$omp end parallel
end subroutine test01


subroutine test02()
  integer::va
  equivalence (va,vva)
  namelist /na1/vva
  va=1

!ERROR: Variable 'va' in NAMELIST cannot be in a PRIVATE clause
!$omp parallel private(va)
  write(*,na1)
!$omp end parallel
end subroutine test02
