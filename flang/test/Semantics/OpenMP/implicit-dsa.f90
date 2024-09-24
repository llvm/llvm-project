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
    !DEF: /implicit_dsa_test1/OtherConstruct1/y (OmpPrivate) HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct1/z (OmpShared) HostAssoc INTEGER(4)
    x = y + z
  !$omp end task

  !$omp task default(shared)
    !DEF: /implicit_dsa_test1/OtherConstruct2/x HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct2/y HostAssoc INTEGER(4)
    !DEF: /implicit_dsa_test1/OtherConstruct2/z HostAssoc INTEGER(4)
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
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct1/x HostAssoc INTEGER(4)
      x = 1
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct1/y HostAssoc INTEGER(4)
      y = 1
    !$omp end task

    !$omp task firstprivate(x)
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct2/x (OmpFirstPrivate) HostAssoc INTEGER(4)
      x = 1
      !DEF: /implicit_dsa_test3/OtherConstruct1/OtherConstruct2/z HostAssoc INTEGER(4)
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
        !DEF: /implicit_dsa_test5/OtherConstruct1/OtherConstruct1/OtherConstruct1/x HostAssoc INTEGER(4)
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
      !DEF: /implicit_dsa_test6/OtherConstruct1/OtherConstruct2/y (OmpShared) HostAssoc INTEGER(4)
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
