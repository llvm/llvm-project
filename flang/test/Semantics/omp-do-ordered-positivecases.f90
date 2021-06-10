!RUN: %S/test_errors.sh %s %t %flang -fopenmp
!REQUIRES: shell
! OpenMP Version 4.5
! 2.7.1 Ordered Clause positive cases.

!DEF: /omp_doordered MainProgram
program omp_doordered
  !DEF: /omp_doordered/i ObjectEntity INTEGER(4)
  !DEF: /omp_doordered/j ObjectEntity INTEGER(4)
  integer i, j
  !$omp do  ordered(2)
  !DEF: /omp_doordered/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_doordered/Block1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=1,10
      print *, "hello"
    end do
  end do
  !$omp end do

  !REF: /omp_doordered/i
  do i=1,10
    !REF: /omp_doordered/j
    do j=1,10
      !$omp do  ordered(1)
      !DEF: /omp_doordered/Block2/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      do k=1,10
        print *, "hello"
      end do
      !$omp end do
    end do
  end do

  !$omp do  ordered(1)
  !DEF: /omp_doordered/Block3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !$omp ordered
    !REF: /omp_doordered/j
    do j=1,10
      print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end do

  !$omp do  collapse(1) ordered(2)
  !DEF: /omp_doordered/Block4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_doordered/Block4/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=1,10
      print *, "hello"
    end do
  end do
  !$omp end do

  !$omp parallel  num_threads(4)
  !$omp do  ordered(1) collapse(1)
  !DEF: /omp_doordered/Block5/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !$omp ordered
    !DEF: /omp_doordered/Block5/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=1,10
      print *, "hello"
    end do
    !$omp end ordered
  end do
  !$omp end parallel
end program omp_doordered
