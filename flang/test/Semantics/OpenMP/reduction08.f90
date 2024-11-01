! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause Positive cases

!DEF: /omp_reduction MainProgram
program omp_reduction
  !DEF: /omp_reduction/i ObjectEntity INTEGER(4)
  integer i
  !DEF: /omp_reduction/k ObjectEntity INTEGER(4)
  integer :: k = 10
  !DEF: /omp_reduction/m ObjectEntity INTEGER(4)
  integer :: m = 12
  !$omp parallel do  reduction(max:k)
  !DEF: /omp_reduction/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct1/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/max ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /omp_reduction/OtherConstruct1/m HostAssoc INTEGER(4)
    k = max(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(min:k)
  !DEF: /omp_reduction/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct2/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/min ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /omp_reduction/OtherConstruct2/m HostAssoc INTEGER(4)
    k = min(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(iand:k)
  !DEF: /omp_reduction/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct3/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/iand ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /omp_reduction/OtherConstruct3/m HostAssoc INTEGER(4)
    k = iand(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ior:k)
  !DEF: /omp_reduction/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct4/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/ior ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /omp_reduction/OtherConstruct4/m HostAssoc INTEGER(4)
    k = ior(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ieor:k)
  !DEF: /omp_reduction/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct5/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/ieor ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !DEF: /omp_reduction/OtherConstruct5/m HostAssoc INTEGER(4)
    k = ieor(k,m)
  end do
  !$omp end parallel do

end program omp_reduction
