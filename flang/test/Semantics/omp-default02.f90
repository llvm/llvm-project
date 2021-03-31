!RUN: %S/test_errors.sh %s %t %flang -fopenmp
! OpenMP Version 4.5
! 2.15.3.1 default Clause - a positive test case.

!DEF: /omp_default MainProgram
program omp_default
 !DEF: /omp_default/a ObjectEntity INTEGER(4)
 !DEF: /omp_default/b ObjectEntity INTEGER(4)
 !DEF: /omp_default/c ObjectEntity INTEGER(4)
 !DEF: /omp_default/i ObjectEntity INTEGER(4)
 !DEF: /omp_default/k ObjectEntity INTEGER(4)
 integer a(10), b(10), c(10), i, k
!$omp parallel  default(shared)
 !DEF: /omp_default/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,10
  !REF: /omp_default/c
  !REF: /omp_default/Block1/i
  !REF: /omp_default/a
  !REF: /omp_default/b
  !REF: /omp_default/k
  c(i) = a(i)+b(i)+k
 end do
!$omp end parallel
!$omp task  default(shared)
 !DEF: /omp_default/Block2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,10
  !REF: /omp_default/c
  !REF: /omp_default/Block2/i
  !REF: /omp_default/a
  !REF: /omp_default/b
  !REF: /omp_default/k
  c(i) = a(i)+b(i)+k
 end do
!$omp end task
!$omp taskloop  default(shared)
 !DEF: /omp_default/Block3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 do i=1,10
  !REF: /omp_default/c
  !REF: /omp_default/Block3/i
  !REF: /omp_default/a
  !REF: /omp_default/b
  !REF: /omp_default/k
  c(i) = a(i)+b(i)+k
 end do
!$omp end taskloop
!$omp teams  default(shared)
 !REF: /omp_default/i
 do i=1,10
  !REF: /omp_default/c
  !REF: /omp_default/i
  !REF: /omp_default/a
  !REF: /omp_default/b
  !REF: /omp_default/k
  c(i) = a(i)+b(i)+k
 end do
!$omp end teams
end program omp_default
