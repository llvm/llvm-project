! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.15.3.6 Reduction Clause Positive cases.
!DEF: /omp_reduction MainProgram
program omp_reduction
  !DEF: /omp_reduction/i ObjectEntity INTEGER(4)
  integer i
  !DEF: /omp_reduction/k ObjectEntity INTEGER(4)
  integer :: k = 10
  !DEF: /omp_reduction/a ObjectEntity INTEGER(4)
  integer a(10)
  !DEF: /omp_reduction/b ObjectEntity INTEGER(4)
  integer b(10,10,10)

  !$omp parallel  shared(k)
  !$omp do  reduction(+:k)
  !DEF: /omp_reduction/OtherConstruct1/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct1/OtherConstruct1/k (OmpReduction) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do
  !$omp end parallel


  !$omp parallel do  reduction(+:a(10))
  !DEF: /omp_reduction/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_reduction/k
    k = k+1
  end do
  !$omp end parallel do


  !$omp parallel do  reduction(+:a(1:10:1))
  !DEF: /omp_reduction/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_reduction/k
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2))
  !DEF: /omp_reduction/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_reduction/k
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel do  reduction(+:b(1:10:1,1:5,2:5:1))
  !DEF: /omp_reduction/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !REF: /omp_reduction/k
    k = k+1
  end do
  !$omp end parallel do

  !$omp parallel  private(i)
  !$omp do reduction(+:k) reduction(+:j)
  !DEF: /omp_reduction/OtherConstruct6/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct6/OtherConstruct1/k (OmpReduction) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do
  !$omp end parallel

  !$omp do reduction(-:k) reduction(*:j) reduction(-:l)
  !DEF: /omp_reduction/OtherConstruct7/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct7/k (OmpReduction) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do


  !$omp do reduction(.and.:k) reduction(.or.:j) reduction(.eqv.:l)
  !DEF: /omp_reduction/OtherConstruct8/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_reduction/OtherConstruct8/k (OmpReduction) HostAssoc INTEGER(4)
    k = k+1
  end do
  !$omp end do

end program omp_reduction
