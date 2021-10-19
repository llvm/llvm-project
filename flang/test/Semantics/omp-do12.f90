! RUN: %python %S/test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop constructs.

!DEF: /omp_cycle MainProgram
program omp_cycle
  !$omp do  collapse(1)
  !DEF: /omp_cycle/Block1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !REF: /omp_cycle/Block1/i
    if (i<1) cycle
    !DEF: /omp_cycle/j (Implicit) ObjectEntity INTEGER(4)
    do j=0,10
      !DEF: /omp_cycle/k (Implicit) ObjectEntity INTEGER(4)
      do k=0,10
        !REF: /omp_cycle/Block1/i
        !REF: /omp_cycle/j
        !REF: /omp_cycle/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(1)
  !DEF: /omp_cycle/Block2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !REF: /omp_cycle/j
    do j=0,10
      !REF: /omp_cycle/Block2/i
      if (i<1) cycle
      !REF: /omp_cycle/k
      do k=0,10
        !REF: /omp_cycle/Block2/i
        !REF: /omp_cycle/j
        !REF: /omp_cycle/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(2)
  !DEF: /omp_cycle/Block3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !DEF: /omp_cycle/Block3/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=0,10
      !REF: /omp_cycle/k
      do k=0,10
        !REF: /omp_cycle/Block3/i
        if (i<1) cycle
        !REF: /omp_cycle/Block3/i
        !REF: /omp_cycle/Block3/j
        !REF: /omp_cycle/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(3)
  !DEF: /omp_cycle/Block4/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  do i=0,10
    !DEF: /omp_cycle/Block4/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do j=0,10
      !DEF: /omp_cycle/Block4/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      do k=0,10
        !REF: /omp_cycle/Block4/i
        if (i<1) cycle
        !REF: /omp_cycle/Block4/i
        !REF: /omp_cycle/Block4/j
        !REF: /omp_cycle/Block4/k
        print *, i, j, k
      end do
    end do
  end do
  !$omp end do

  !$omp do  collapse(3)
  !DEF: /omp_cycle/Block5/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo:do i=0,10
    !DEF: /omp_cycle/Block5/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    foo1:do j=0,10
      !DEF: /omp_cycle/Block5/k (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      foo2:do k=0,10
        !REF: /omp_cycle/Block5/i
        if (i<1) cycle foo2
        !REF: /omp_cycle/Block5/i
        !REF: /omp_cycle/Block5/j
        !REF: /omp_cycle/Block5/k
        print *, i, j, k
      end do foo2
    end do foo1
  end do foo
  !$omp end do
end program omp_cycle
