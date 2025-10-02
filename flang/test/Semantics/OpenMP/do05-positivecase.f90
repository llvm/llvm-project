! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct restrictions on single directive.
! A positive case

!DEF: /OMP_DO MainProgram
program OMP_DO
  !DEF: /OMP_DO/i ObjectEntity INTEGER(4)
  !DEF: /OMP_DO/n ObjectEntity INTEGER(4)
  integer i,n
  !$omp parallel
  !DEF: /OMP_DO/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !$omp single
    print *, "hello"
    !$omp end single
  end do
  !$omp end parallel

  !$omp parallel  default(shared)
  !$omp do
  !DEF: /OMP_DO/OtherConstruct2/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  !DEF: /OMP_DO/OtherConstruct2/OtherConstruct1/n HostAssoc INTEGER(4)
  do i=1,n
    !$omp parallel
    !$omp single
    !DEF: /work EXTERNAL (Subroutine) ProcEntity
    !DEF: /OMP_DO/OtherConstruct2/OtherConstruct1/OtherConstruct1/OtherConstruct1/i HostAssoc INTEGER(4)
    call work(i, 1)
    !$omp end single
    !$omp end parallel
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel private(i)
  !DEF: /OMP_DO/OtherConstruct3/i (OmpPrivate, OmpExplicit) HostAssoc INTEGER(4)
  do i=1,10
     !$omp single
     print *, "hello"
     !$omp end single
  end do
  !$omp end parallel

  !$omp target teams distribute parallel do
  !DEF:/OMP_DO/OtherConstruct4/i (OmpPrivate ,OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF:/OMP_DO/OtherConstruct4/i
    if(i<10) cycle
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do simd
  !DEF:/OMP_DO/OtherConstruct5/i (OmpLinear,OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF:/OMP_DO/OtherConstruct5/i
    if(i<10) cycle
  end do
  !$omp end target teams distribute parallel do simd

  !$omp target teams distribute 
  !DEF: /OMP_DO/OtherConstruct6/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF: /OMP_DO/OtherConstruct6/i
    if(i < 5) cycle
  end do

  !$omp target teams distribute simd
  !DEF: /OMP_DO/OtherConstruct7/i (OmpLinear, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF: /OMP_DO/OtherConstruct7/i
    if(i < 5) cycle
  end do
end program OMP_DO
