! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause Positive cases

!DEF: /OMP_REDUCTION MainProgram
program OMP_REDUCTION
  !DEF: /OMP_REDUCTION/i ObjectEntity INTEGER(4)
  integer i
  !DEF: /OMP_REDUCTION/k ObjectEntity INTEGER(4)
  integer :: k = 10
  !DEF: /OMP_REDUCTION/m ObjectEntity INTEGER(4)
  integer :: m = 12
  !$omp parallel do  reduction(max:k)
  !DEF: /OMP_REDUCTION/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct1/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /OMP_REDUCTION/max ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /OMP_REDUCTION/OtherConstruct1/m (OmpShared) HostAssoc INTEGER(4)
    k = max(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(min:k)
  !DEF: /OMP_REDUCTION/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct2/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /OMP_REDUCTION/min ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /OMP_REDUCTION/OtherConstruct2/m (OmpShared) HostAssoc INTEGER(4)
    k = min(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(iand:k)
  !DEF: /OMP_REDUCTION/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct3/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /OMP_REDUCTION/iand ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /OMP_REDUCTION/OtherConstruct3/m (OmpShared) HostAssoc INTEGER(4)
    k = iand(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ior:k)
  !DEF: /OMP_REDUCTION/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct4/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /OMP_REDUCTION/ior ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /OMP_REDUCTION/OtherConstruct4/m (OmpShared) HostAssoc INTEGER(4)
    k = ior(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ieor:k)
  !DEF: /OMP_REDUCTION/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct5/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /OMP_REDUCTION/ieor ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /OMP_REDUCTION/OtherConstruct5/m (OmpShared) HostAssoc INTEGER(4)
    k = ieor(k,m)
  end do
  !$omp end parallel do

end program OMP_REDUCTION
