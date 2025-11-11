! RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s | FileCheck %s
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause Positive cases.
!DEF: /OMP_REDUCTION MainProgram
program OMP_REDUCTION
  integer i
  integer :: k = 10
  integer a(10)
  integer b(10,10,10)

  !$omp parallel  shared(k)
  !$omp do  reduction(+:k)
  do i=1,10
    k = k+1
  end do
  !$omp end do
  !$omp end parallel


  !$omp parallel do  reduction(+:a(10))
  do i=1,10
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:a(10))
  do i=1,10
    a(10) = a(10)+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:a(1:10:1))
  do i=1,10
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2))
  do i=1,10
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2:5:1))
  do i=1,10
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel  private(i)
  !$omp do reduction(+:k) reduction(+:j)
  do i=1,10
    k = k+1
  end do
  !$omp end do
  !$omp end parallel

  !$omp do reduction(+:k) reduction(*:j) reduction(+:l)
  do i=1,10
    k = k+1
  end do
  !$omp end do
end program OMP_REDUCTION

! CHECK: !DEF: /OMP_REDUCTION MainProgram
! CHECK-NEXT: program OMP_REDUCTION
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/i ObjectEntity INTEGER(4)
! CHECK-NEXT:  integer i
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/k ObjectEntity INTEGER(4)
! CHECK-NEXT:  integer :: k = 10
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/a ObjectEntity INTEGER(4)
! CHECK-NEXT:  integer a(10)
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/b ObjectEntity INTEGER(4)
! CHECK-NEXT:  integer b(10,10,10)
! CHECK-NEXT: !$omp parallel shared(k)
! CHECK-NEXT: !$omp do reduction(+: k)
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct1/OtherConstruct1/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end do
! CHECK-NEXT: !$omp end parallel
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/reduction_temp_a(10) (CompilerCreated) ObjectEntity INTEGER(4)
! CHECK-NEXT:  !REF: /OMP_REDUCTION/a
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct3/a (OmpShared) HostAssoc INTEGER(4)
! CHECK-NEXT:  reduction_temp_a(10) = a(10)
! CHECK-NEXT: !$omp parallel do reduction(+: reduction_temp_a(10))
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct2/k (OmpShared) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end parallel do
! CHECK-NEXT:  !REF: /OMP_REDUCTION/reduction_temp_a(10)
! CHECK-NEXT:  !REF: /OMP_REDUCTION/a
! CHECK-NEXT:  !REF: /OMP_REDUCTION/OtherConstruct3/a
! CHECK-NEXT:  reduction_temp_a(10) = a(10)
! CHECK-NEXT: !$omp parallel do reduction(+: reduction_temp_a(10))
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !REF: /OMP_REDUCTION/reduction_temp_a(10)
! CHECK-NEXT:   reduction_temp_a(10) = reduction_temp_a(10)+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end parallel do
! CHECK-NEXT:  !REF: /OMP_REDUCTION/reduction_temp_a(10)
! CHECK-NEXT:  !REF: /OMP_REDUCTION/a
! CHECK-NEXT:  !REF: /OMP_REDUCTION/OtherConstruct3/a
! CHECK-NEXT:  a(10) = reduction_temp_a(10)
! CHECK-NEXT: !$omp parallel do reduction(+: a(1:10:1))
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct4/k (OmpShared) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end parallel do
! CHECK-NEXT: !$omp parallel do reduction(+: b(1:10:1,1:5,2))
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct5/k (OmpShared) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end parallel do
! CHECK-NEXT: !$omp parallel do reduction(+: b(1:10:1,1:5,2:5:1))
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct6/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct6/k (OmpShared) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end parallel do
! CHECK-NEXT: !$omp parallel private(i)
! CHECK-NEXT: !$omp do reduction(+: k) reduction(+: j)
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct7/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct7/OtherConstruct1/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end do
! CHECK-NEXT: !$omp end parallel
! CHECK-NEXT: !$omp do reduction(+: k) reduction(*: j) reduction(+: l)
! CHECK-NEXT:  !DEF: /OMP_REDUCTION/OtherConstruct8/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
! CHECK-NEXT:  do i=1,10
! CHECK-NEXT:   !DEF: /OMP_REDUCTION/OtherConstruct8/k (OmpReduction, OmpExplicit) HostAssoc INTEGER(4)
! CHECK-NEXT:   k = k+1
! CHECK-NEXT:  end do
! CHECK-NEXT: !$omp end do
! CHECK-NEXT: end program OMP_REDUCTION