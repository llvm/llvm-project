! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop constructs.

!DEF: /OMP_CYCLE MainProgram
program OMP_CYCLE
  !$omp do  collapse(1)
  !DEF: /OMP_CYCLE/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    cycle
    !DEF: /OMP_CYCLE/j (Implicit) ObjectEntity INTEGER(4)
    do j=0,10
      !DEF: /OMP_CYCLE/k (Implicit) ObjectEntity INTEGER(4)
      do k=0,10
        !REF: /OMP_CYCLE/OtherConstruct1/i
        !REF: /OMP_CYCLE/j
        !REF: /OMP_CYCLE/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(1)
  !DEF: /OMP_CYCLE/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !REF: /OMP_CYCLE/j
    do j=0,10
      cycle
      !REF: /OMP_CYCLE/k
      do k=0,10
        !REF: /OMP_CYCLE/OtherConstruct2/i
        !REF: /OMP_CYCLE/j
        !REF: /OMP_CYCLE/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(2)
  !DEF: /OMP_CYCLE/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !DEF: /OMP_CYCLE/OtherConstruct3/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=0,10
      !REF: /OMP_CYCLE/k
      do k=0,10
        cycle
        !REF: /OMP_CYCLE/OtherConstruct3/i
        !REF: /OMP_CYCLE/OtherConstruct3/j
        !REF: /OMP_CYCLE/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(3)
  !DEF: /OMP_CYCLE/OtherConstruct4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !DEF: /OMP_CYCLE/OtherConstruct4/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=0,10
      !DEF: /OMP_CYCLE/OtherConstruct4/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      do k=0,10
        cycle
        !REF: /OMP_CYCLE/OtherConstruct4/i
        !REF: /OMP_CYCLE/OtherConstruct4/j
        !REF: /OMP_CYCLE/OtherConstruct4/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  ordered(3)
  !DEF: /OMP_CYCLE/OtherConstruct5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo:do i=0,10
    !DEF: /OMP_CYCLE/OtherConstruct5/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    foo1:do j=0,10
      !DEF: /OMP_CYCLE/OtherConstruct5/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      foo2:do k=0,10
        cycle foo2
        !REF: /OMP_CYCLE/OtherConstruct5/i
        !REF: /OMP_CYCLE/OtherConstruct5/j
        !REF: /OMP_CYCLE/OtherConstruct5/k
        print *, i, j, k
      end do foo2
    end do foo1
  end do foo
  !$omp end do
end program OMP_CYCLE
