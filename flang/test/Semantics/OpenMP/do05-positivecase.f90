! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Loop Construct restrictions on single directive.
! A positive case

!DEF: /omp_do MainProgram
program omp_do
  !DEF: /omp_do/i ObjectEntity INTEGER(4)
  !DEF: /omp_do/n ObjectEntity INTEGER(4)
  integer i,n
  !$omp parallel
  !DEF: /omp_do/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !$omp single
    print *, "hello"
    !$omp end single
  end do
  !$omp end parallel

  !$omp parallel  default(shared)
  !$omp do
  !DEF: /omp_do/OtherConstruct2/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  !DEF: /omp_do/OtherConstruct2/n HostAssoc INTEGER(4)
  do i=1,n
    !$omp parallel
    !$omp single
    !DEF: /work EXTERNAL (Subroutine) ProcEntity
    !DEF: /omp_do/OtherConstruct2/OtherConstruct1/OtherConstruct1/i HostAssoc INTEGER(4)
    call work(i, 1)
    !$omp end single
    !$omp end parallel
  end do
  !$omp end do
  !$omp end parallel

  !$omp parallel private(i)
  !DEF: /omp_do/OtherConstruct3/i (OmpPrivate) HostAssoc INTEGER(4)
  do i=1,10
     !$omp single
     print *, "hello"
     !$omp end single
  end do
  !$omp end parallel

  !$omp target teams distribute parallel do
  !DEF:/omp_do/OtherConstruct4/i (OmpPrivate ,OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF:/omp_do/OtherConstruct4/i
    if(i<10) cycle
  end do
  !$omp end target teams distribute parallel do

  !$omp target teams distribute parallel do simd
  !DEF:/omp_do/OtherConstruct5/i (OmpLinear,OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,100
    !REF:/omp_do/OtherConstruct5/i
    if(i<10) cycle
  end do
  !$omp end target teams distribute parallel do simd
end program omp_do
