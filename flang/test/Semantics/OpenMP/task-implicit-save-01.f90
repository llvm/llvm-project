! RUN: %python %S/../test_symbols.py %s %flang_fc1 -fopenmp

! Variables with static storage duration (SAVE attribute, explicit or implied,
! or in a common block) are shared in the enclosing context. Unless an earlier
! rule determines their data-sharing attribute, they are also implicitly shared
! in task generating constructs (OpenMP 6.0, 7.1.1).
!
! See https://github.com/llvm/llvm-project/issues/201513

!DEF: /MAIN MainProgram
program MAIN
  !DEF: /MAIN/a ObjectEntity INTEGER(4)
  integer :: a = 1
  !DEF: /MAIN/b (InDataStmt) ObjectEntity INTEGER(4)
  integer b
  !REF: /MAIN/b
  data b/1/
  !DEF: /MAIN/c SAVE ObjectEntity INTEGER(4)
  integer, save :: c

  !$omp task
    !DEF: /MAIN/OtherConstruct1/a (OmpShared) HostAssoc INTEGER(4)
    a = 12
    !DEF: /MAIN/OtherConstruct1/b (OmpShared) HostAssoc INTEGER(4)
    b = 13
    !DEF: /MAIN/OtherConstruct1/c (OmpShared) HostAssoc INTEGER(4)
    c = 14
  !$omp end task
end program

!DEF: /test_initialized (Subroutine) Subprogram
subroutine test_initialized
  !DEF: /test_initialized/a ObjectEntity INTEGER(4)
  integer :: a = 1
  !DEF: /test_initialized/b (InDataStmt) ObjectEntity INTEGER(4)
  integer b
  !REF: /test_initialized/b
  data b/1/
  !DEF: /test_initialized/c SAVE ObjectEntity INTEGER(4)
  integer, save :: c
  !DEF: /test_initialized/d (InCommonBlock) ObjectEntity INTEGER(4)
  integer d
  !DEF: /test_initialized/e ObjectEntity INTEGER(4)
  integer e
  !DEF: /test_initialized/blk CommonBlockDetails
  !REF: /test_initialized/d
  common /blk/ d

  !$omp task
    !DEF: /test_initialized/OtherConstruct1/a (OmpShared) HostAssoc INTEGER(4)
    a = 12
    !DEF: /test_initialized/OtherConstruct1/b (OmpShared) HostAssoc INTEGER(4)
    b = 13
    !DEF: /test_initialized/OtherConstruct1/c (OmpShared) HostAssoc INTEGER(4)
    c = 14
    !DEF: /test_initialized/OtherConstruct1/d (OmpShared) HostAssoc INTEGER(4)
    d = 15
    !DEF: /test_initialized/OtherConstruct1/e (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
    e = 16
  !$omp end task
end subroutine

!DEF: /test_bare_save (Subroutine) Subprogram
subroutine test_bare_save
  !DEF: /test_bare_save/x ObjectEntity INTEGER(4)
  integer x
  save

  !$omp task
    !DEF: /test_bare_save/OtherConstruct1/x (OmpShared) HostAssoc INTEGER(4)
    x = 1
  !$omp end task
end subroutine

!DEF: /test_private_in_enclosing (Subroutine) Subprogram
subroutine test_private_in_enclosing
  !DEF: /test_private_in_enclosing/a ObjectEntity INTEGER(4)
  integer :: a = 1

  !$omp parallel private(a)
    !$omp task
      !DEF: /test_private_in_enclosing/OtherConstruct1/OtherConstruct1/a (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
      a = 2
    !$omp end task
  !$omp end parallel
end subroutine

!DEF: /test_nested_tasks (Subroutine) Subprogram
subroutine test_nested_tasks
  !DEF: /test_nested_tasks/a ObjectEntity INTEGER(4)
  integer :: a = 1

  !$omp task
    !$omp task
      !DEF: /test_nested_tasks/OtherConstruct1/OtherConstruct1/a (OmpShared) HostAssoc INTEGER(4)
      a = 2
    !$omp end task
  !$omp end task
end subroutine

!DEF: /test_taskloop (Subroutine) Subprogram
subroutine test_taskloop
  !DEF: /test_taskloop/a ObjectEntity INTEGER(4)
  integer :: a = 1
  !DEF: /test_taskloop/i ObjectEntity INTEGER(4)
  integer i

  !$omp taskloop
    !DEF: /test_taskloop/OtherConstruct1/i (OmpPrivate, OmpPreDetermined) HostAssoc INTEGER(4)
    do i = 1, 10
      !DEF: /test_taskloop/OtherConstruct1/a (OmpShared) HostAssoc INTEGER(4)
      !REF: /test_taskloop/OtherConstruct1/i
      a = a + i
    end do
  !$omp end taskloop
end subroutine
