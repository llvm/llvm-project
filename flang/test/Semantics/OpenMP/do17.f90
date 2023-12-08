! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp
! OpenMP Version 4.5
! 2.7.1 Do Loop constructs.

!DEF: /test MainProgram
program test
 !DEF: /test/i ObjectEntity INTEGER(4)
 !DEF: /test/j ObjectEntity INTEGER(4)
 !DEF: /test/k ObjectEntity INTEGER(4)
 integer i, j, k
 !$omp do  collapse(2)
 !DEF: /test/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 foo: do i=0,10
  !DEF: /test/OtherConstruct1/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo1: do j=0,10
   !REF: /test/k
   foo2: do k=0,10
    !REF: /test/OtherConstruct1/i
    select case (i)
    case (5)
     cycle foo1
    case (7)
     cycle foo2
    end select
    !REF: /test/OtherConstruct1/i
    !REF: /test/OtherConstruct1/j
    !REF: /test/k
    print *, i, j, k
   end do foo2
  end do foo1
 end do foo

 !$omp do  collapse(2)
 !DEF: /test/OtherConstruct2/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
 foo: do i=0,10
  !DEF: /test/OtherConstruct2/j (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
  foo1: do j=0,10
   !REF: /test/k
   foo2: do k=0,10
    !REF: /test/OtherConstruct2/i
    if (i<3) then
     cycle foo1
     !REF: /test/OtherConstruct2/i
    else if (i>8) then
     cycle foo1
    else
     cycle foo2
    end if
    !REF: /test/OtherConstruct2/i
    !REF: /test/OtherConstruct2/j
    !REF: /test/k
    print *, i, j, k
   end do foo2
  end do foo1
 end do foo
!$omp end do
end program test
