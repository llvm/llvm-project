! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! Test symbols generated in block constructs that have implicitly
! determined DSAs.

! Basic cases.
!DEF: /implicit_dsa_test1 (Subroutine) Subprogram
subroutine implicit_dsa_test1
  !DEF: /implicit_dsa_test1/i ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test1/x ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test1/y ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test1/z ObjectEntity INTEGER(4)
  integer i, x, y, z

  !$omp task private(y) shared(z)
    !DEF: /implicit_dsa_test1/OtherConstruct1/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct1/y (OmpPrivate, OmpExplicit) HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct1/z (OmpShared, OmpExplicit) HostAssoc INTEGER(4)
    x = y + z
  !$omp end task

  !$omp task default(shared)
    !DEF: /implicit_dsa_test1/OtherConstruct2/x (OmpShared) HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct2/y (OmpShared) HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct2/z (OmpShared) HostAssoc INTEGER(4)
    x = y + z
  !$omp end task

  !$omp taskloop
    !DEF: /implicit_dsa_test1/OtherConstruct3/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do i = 0, 10
      !DEF: /implicit_dsa_test1/OtherConstruct3/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      !DEF: /implicit_dsa_test1/OtherConstruct3/y (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      !REF: /implicit_dsa_test1/OtherConstruct3/i
      x = y + i
    end do
  !$omp end taskloop
end subroutine

! Nested task with implicit firstprivate DSA variable.
!DEF: /implicit_dsa_test2 (Subroutine) Subprogram
subroutine implicit_dsa_test2
  !DEF: /implicit_dsa_test2/x ObjectEntity INTEGER(4)
  integer x

  !$omp task
    !$omp task
      !DEF: /implicit_dsa_test2/OtherConstruct1/OtherConstruct1/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      x = 1
    !$omp end task
  !$omp end task
end subroutine

! Nested tasks with implicit shared DSA variables.
!DEF: /implicit_dsa_test3 (Subroutine) Subprogram
subroutine implicit_dsa_test3
  !DEF: /implicit_dsa_test3/x ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test3/y ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test3/z ObjectEntity INTEGER(4)
  integer x, y, z

  !$omp parallel
    !$omp task
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct1/x (OmpShared) HostAssoc INTEGER(4)
      x = 1
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct1/y (OmpShared) HostAssoc INTEGER(4)
      y = 1
    !$omp end task

    !$omp task firstprivate(x)
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct2/x (OmpFirstPrivate, OmpExplicit) HostAssoc INTEGER(4)
      x = 1
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct2/z (OmpShared) HostAssoc INTEGER(4)
      z = 1
    !$omp end task
  !$omp end parallel
end subroutine

! Task with implicit firstprivate DSA variables, enclosed in private context.
!DEF: /implicit_dsa_test4 (Subroutine) Subprogram
subroutine implicit_dsa_test4
  !DEF: /implicit_dsa_test4/x ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test4/y ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test4/z ObjectEntity INTEGER(4)
  integer x, y, z

  !$omp parallel default(private)
    !$omp task
      !DEF: /implicit_dsa_test4/OtherConstruct1/OtherConstruct1/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      x = 0
      !DEF: /implicit_dsa_test4/OtherConstruct1/OtherConstruct1/z (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      z = 1
    !$omp end task

    !$omp task
      !DEF: /implicit_dsa_test4/OtherConstruct1/OtherConstruct2/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      x = 1
      !DEF: /implicit_dsa_test4/OtherConstruct1/OtherConstruct2/y (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      y = 0
    !$omp end task
  !$omp end parallel
end subroutine

! Inner parallel using implicit firstprivate symbol.
!DEF: /implicit_dsa_test5 (Subroutine) Subprogram
subroutine implicit_dsa_test5
  !DEF: /implicit_dsa_test5/x ObjectEntity INTEGER(4)
  integer x

  !$omp parallel default(private)
    !$omp task
      !$omp parallel
        !DEF: /implicit_dsa_test5/OtherConstruct1/OtherConstruct1/OtherConstruct1/x (OmpShared) HostAssoc INTEGER(4)
        x = 1
      !$omp end parallel
    !$omp end task
  !$omp end parallel
end subroutine

! Constructs nested inside a task with implicit DSA variables.
!DEF: /implicit_dsa_test6 (Subroutine) Subprogram
subroutine implicit_dsa_test6
  !DEF: /implicit_dsa_test6/x ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test6/y ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test6/z ObjectEntity INTEGER(4)
  integer x, y, z

  !$omp task
    !$omp parallel default(private)
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct1/x (OmpPrivate) HostAssoc INTEGER(4)
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct1/y (OmpPrivate) HostAssoc INTEGER(4)
      x = y
    !$omp end parallel

    !$omp parallel default(firstprivate) shared(y)
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct2/y (OmpShared, OmpExplicit) HostAssoc INTEGER(4)
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct2/x (OmpFirstPrivate) HostAssocINTEGER(4)
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct2/z (OmpFirstPrivate) HostAssocINTEGER(4)
      y = x + z
    !$omp end parallel
  !$omp end task
end subroutine

! Test taskgroup - it uses the same scope as task.
!DEF: /implicit_dsa_test7 (Subroutine) Subprogram
subroutine implicit_dsa_test7
  !DEF: /implicit_dsa_test7/x ObjectEntity INTEGER(4)
  !DEF: /implicit_dsa_test7/y ObjectEntity INTEGER(4)
  integer x, y

  !$omp task
    !$omp taskgroup
      !DEF: /implicit_dsa_test7/OtherConstruct1/x (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      !DEF: /implicit_dsa_test7/OtherConstruct1/y (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      x = y
    !$omp end taskgroup
  !$omp end task
end subroutine

! Predetermined loop iteration variable.
!DEF: /implicit_dsa_test8 (Subroutine) Subprogram
subroutine implicit_dsa_test8
  !DEF: /implicit_dsa_test8/i ObjectEntity INTEGER(4)
  integer i

  !$omp task
    !DEF: /implicit_dsa_test8/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do i = 1, 10
    end do
  !$omp end task
end subroutine

! Test variables defined in modules default to shared DSA
!DEF: /implicit_dsa_test9_mod Module
module implicit_dsa_test9_mod
 !DEF: /implicit_dsa_test9_mod/tm3a PUBLIC (InDataStmt) ObjectEntity COMPLEX(4)
  complex tm3a/(0,0)/
 !DEF: /implicit_dsa_test9_mod/tm4a PUBLIC ObjectEntity COMPLEX(4)
  complex tm4a
contains
 !DEF: /implicit_dsa_test9_mod/implict_dsa_test9 PUBLIC (Subroutine) Subprogram
  subroutine implict_dsa_test9
    !$omp task
      !$omp task
        !DEF: /implicit_dsa_test9_mod/implict_dsa_test9/OtherConstruct1/OtherConstruct1/tm3a (OmpShared) HostAssoc COMPLEX(4)
        tm3a = (1, 2)
        !DEF: /implicit_dsa_test9_mod/implict_dsa_test9/OtherConstruct1/OtherConstruct1/tm4a (OmpShared) HostAssoc COMPLEX(4)
        tm4a = (3, 4)
      !$omp end task
    !$omp end task
  !$omp taskwait
  !REF: /implicit_dsa_test9_mod/tm3a
  print *,tm3a
  end subroutine
end module

! Test variables in data statement default to shared DSA
!DEF: /implicit_dsa_test10 (Subroutine) Subprogram
subroutine implicit_dsa_test10
 !DEF: /implicit_dsa_test10/tm3a (Implicit, InDataStmt) ObjectEntity REAL(4)
data tm3a /3/
!$omp task
  !$omp task
 !DEF: /implicit_dsa_test10/OtherConstruct1/OtherConstruct1/tm3a (OmpShared) HostAssoc REAL(4)
    tm3a = 5
  !$omp end task
!$omp end task
!$omp taskwait
 !REF: /implicit_dsa_test10/tm3a
print *,tm3a
end subroutine

! Test variables with the SAVE attrtibute default to shared DSA
!DEF: /implicit_dsa_test_11 (Subroutine) Subprogram
subroutine implicit_dsa_test_11
 !DEF: /implicit_dsa_test_11/tm3a SAVE ObjectEntity COMPLEX(4)
complex, save :: tm3a
!$omp task
  !$omp task
    !DEF: /implicit_dsa_test_11/OtherConstruct1/OtherConstruct1/tm3a (OmpShared) HostAssoc COMPLEX(4)
    tm3a = (1, 2)
  !$omp end task
!$omp end task
!$omp taskwait
!REF: /implicit_dsa_test_11/tm3a
print *,tm3a
end subroutine

! Test variables referenced in a common block default to shared DSA
!DEF: /implicit_dsa_test_12 (Subroutine) Subprogram
subroutine implicit_dsa_test_12
 !DEF: /implicit_dsa_test_12/tm3a (InCommonBlock) ObjectEntity COMPLEX(4)
complex tm3a
 !DEF: /implicit_dsa_test_12/tcom CommonBlockDetails
 !REF: /implicit_dsa_test_12/tm3a
common /tcom/ tm3a
!$omp task
  !$omp task
    !DEF: /implicit_dsa_test_12/OtherConstruct1/OtherConstruct1/tm3a (OmpShared) HostAssoc COMPLEX(4)
    tm3a = (1, 2)
  !$omp end task
!$omp end task
!$omp taskwait
!REF: /implicit_dsa_test_12/tm3a
print *,tm3a
end subroutine

! Test static duration variables with DSA set in the enclosing scope do not default to shared DSA
!DEF: /implicit_dsa_test_13_mod Module
module implicit_dsa_test_13_mod
  !DEF: /implicit_dsa_test_13_mod/a PUBLIC ObjectEntity INTEGER(4)
  integer::a=5
contains
  !DEF: /implicit_dsa_test_13_mod/implicit_dsa_test_13 PUBLIC (Subroutine) Subprogram
  subroutine implicit_dsa_test_13
    !DEF: /implicit_dsa_test_13_mod/implicit_dsa_test_13/i ObjectEntity INTEGER(4)
    integer i
    !$omp do private(a)
      !DEF: /implicit_dsa_test_13_mod/implicit_dsa_test_13/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
      do i=0,10
        !$omp task
        !DEF: /implicit_dsa_test_13_mod/implicit_dsa_test_13/OtherConstruct1/OtherConstruct1/a (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
        !DEF: /implicit_dsa_test_13_mod/implicit_dsa_test_13/OtherConstruct1/OtherConstruct1/i (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
        a=a+i
        !$omp end task
      end do
  end subroutine implicit_dsa_test_13
end module implicit_dsa_test_13_mod
