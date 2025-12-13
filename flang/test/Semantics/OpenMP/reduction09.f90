! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause Positive cases.
!DEF: /OMP_REDUCTION MainProgram
program OMP_REDUCTION
  !DEF: /OMP_REDUCTION/i ObjectEntity INTEGER(4)
  integer i
  !DEF: /OMP_REDUCTION/k ObjectEntity INTEGER(4)
  integer :: k = 10
  !DEF: /OMP_REDUCTION/a ObjectEntity INTEGER(4)
  integer a(10)
  !DEF: /OMP_REDUCTION/b ObjectEntity INTEGER(4)
  integer b(10,10,10)

  !$omp parallel  shared(k)
  !$omp do  reduction(+:k)
  !DEF: /OMP_REDUCTION/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct1/OtherConstruct1/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do
  !$omp end parallel


  !$omp parallel do  reduction(+:a(10))
  !DEF: /OMP_REDUCTION/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct2/k (OmpShared) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end parallel do


  !$omp parallel do  reduction(+:a(1:10:1))
  !DEF: /OMP_REDUCTION/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct3/k (OmpShared) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2))
  !DEF: /OMP_REDUCTION/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct4/k (OmpShared) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2:5:1))
  !DEF: /OMP_REDUCTION/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct5/k (OmpShared) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel  private(i)
  !$omp do reduction(+:k) reduction(+:j)
  !DEF: /OMP_REDUCTION/OtherConstruct6/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct6/OtherConstruct1/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do
  !$omp end parallel

  !$omp do reduction(+:k) reduction(*:j) reduction(+:l)
  !DEF: /OMP_REDUCTION/OtherConstruct7/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /OMP_REDUCTION/OtherConstruct7/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do
end program OMP_REDUCTION
