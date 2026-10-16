!RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp -fopenmp-version=52

!DEF: /f00 (Subroutine) Subprogram
subroutine f00
  implicit none

!DEF: /f00/n PARAMETER ObjectEntity INTEGER(4)
  integer, parameter :: n = 1024
 !DEF: /f00/i ObjectEntity INTEGER(4)
 !DEF: /f00/j ObjectEntity INTEGER(4)
 !DEF: /f00/k ObjectEntity INTEGER(4)
 !DEF: /f00/array ObjectEntity INTEGER(4)
 !REF: /f00/n
  integer i, j, k, array(n, n, n)

  !The i and j are predetermined private as loop induction variables nested
  !in a teams construct.
  !$omp target teams distribute default(none) shared(array)
!DEF: /f00/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
!REF: /f00/n
  do i = 1, n
!DEF: /f00/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
!REF: /f00/n
    do j = 1, n
      !i and j are shared in parallel
      !$omp parallel do shared(array)
!DEF: /f00/OtherConstruct1/OtherConstruct1/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
!REF: /f00/n
      do k = 1, n
!DEF: /f00/OtherConstruct1/OtherConstruct1/array (OmpShared, OmpExplicit) HostAssoc INTEGER(4)
!DEF: /f00/OtherConstruct1/OtherConstruct1/i (OmpShared) HostAssoc INTEGER(4)
!DEF: /f00/OtherConstruct1/OtherConstruct1/j (OmpShared) HostAssoc INTEGER(4)
!REF: /f00/OtherConstruct1/OtherConstruct1/k
        array(i, j, k) = i + j + k
      enddo
    enddo
  enddo
end subroutine



