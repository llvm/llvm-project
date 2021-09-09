!RUN: %python %S/test_errors.py %s %flang -fopenmp
! OpenMP Version 4.5
! 2.7.1 Collapse Clause Positive cases

!DEF: /omp_docollapse MainProgram
program omp_docollapse
  !DEF: /omp_docollapse/i ObjectEntity INTEGER(4)
  !DEF: /omp_docollapse/j ObjectEntity INTEGER(4)
  !DEF: /omp_docollapse/k ObjectEntity INTEGER(4)
  integer i, j, k
  !$omp do  collapse(2)
  !DEF: /omp_docollapse/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=1,10
    !DEF: /omp_docollapse/Block1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=1,10
      !REF: /omp_docollapse/k
      do k=1,10
        print *, "hello"
      end do
    end do
  end do
  !$omp end do

  !REF: /omp_docollapse/i
  do i=1,10
  !$omp do  collapse(2)
    !DEF: /omp_docollapse/Block1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=1,10
      !DEF: /omp_docollapse/Block1/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      do k=1,10
        print *, "hello"
      end do
    end do
    !$omp end do
  end do
end program omp_docollapse
