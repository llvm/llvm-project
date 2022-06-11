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
  !DEF: /omp_reduction/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/Block1/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/max ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !REF: /omp_reduction/m
    k = max(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(min:k)
  !DEF: /omp_reduction/Block2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/Block2/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/min ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !REF: /omp_reduction/m
    k = min(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(iand:k)
  !DEF: /omp_reduction/Block3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/Block3/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/iand ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !REF: /omp_reduction/m
    k = iand(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ior:k)
  !DEF: /omp_reduction/Block4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/Block4/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/ior ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !REF: /omp_reduction/m
    k = ior(k, m)
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(ieor:k)
  !DEF: /omp_reduction/Block5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/Block5/k (OmpReduction) HostAssoc INTEGER(4)
    !DEF: /omp_reduction/ieor ELEMENTAL, INTRINSIC, PURE (Function) ProcEntity
    !REF: /omp_reduction/m
    k = ieor(k,m)
  end do
  !$omp end parallel do

end program omp_reduction
