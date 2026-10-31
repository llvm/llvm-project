!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp -fno-automatic %s | FileCheck %s

! With -fno-automatic (LanguageFeature::DefaultSave), every local of a
! non-RECURSIVE subprogram is treated as saved; locals of a RECURSIVE
! subprogram are not.

!CHECK-LABEL: !DEF: /nonrec (Subroutine) Subprogram
!CHECK: subroutine nonrec
!CHECK:  !DEF: /nonrec/x ObjectEntity INTEGER(4)
!CHECK:  integer x
!CHECK:  !REF: /nonrec/x
!CHECK:  x = 1
!CHECK: !$omp task
!CHECK:  !DEF: /nonrec/OtherConstruct1/x (OmpShared) HostAssoc INTEGER(4)
!CHECK:  x = 12
!CHECK: !$omp end task
!CHECK: !$omp taskwait
!CHECK: end subroutine

subroutine nonrec()
  integer :: x
  x = 1
  !$omp task
  x = 12
  !$omp end task
  !$omp taskwait
end subroutine

!CHECK-LABEL: !DEF: /rec RECURSIVE (Subroutine) Subprogram
!CHECK: recursive subroutine rec
!CHECK:  !DEF: /rec/y ObjectEntity INTEGER(4)
!CHECK:  integer y
!CHECK:  !REF: /rec/y
!CHECK:  y = 1
!CHECK: !$omp task
!CHECK:  !DEF: /rec/OtherConstruct1/y (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
!CHECK:  y = 12
!CHECK: !$omp end task
!CHECK: !$omp taskwait
!CHECK: end subroutine

recursive subroutine rec()
  integer :: y
  y = 1
  !$omp task
  y = 12
  !$omp end task
  !$omp taskwait
end subroutine
