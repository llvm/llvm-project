!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s | FileCheck %s

!CHECK-LABEL:  !DEF: /block01 (Subroutine) Subprogram
!CHECK: subroutine block01
!CHECK: !$omp parallel default(private)
!CHECK:  block
!CHECK:   !DEF: /block01/OtherConstruct1/BlockConstruct1/cnt ObjectEntity INTEGER(4)
!CHECK:   integer :: cnt = 0
!CHECK:   !REF: /block01/OtherConstruct1/BlockConstruct1/cnt
!CHECK:   cnt = cnt+1
!CHECK:  end block
!CHECK: !$omp end parallel
!CHECK: end subroutine

subroutine block01()
  !$omp parallel default(private)
  block
    integer :: cnt = 0
    cnt = cnt + 1
  end block
  !$omp end parallel
end subroutine

!CHECK-LABEL: !DEF: /block02 (Subroutine) Subprogram
!CHECK: subroutine block02
!CHECK: !$omp parallel default(private)
!CHECK:  block
!CHECK:   !DEF: /block02/OtherConstruct1/BlockConstruct1/a ObjectEntity INTEGER(4)
!CHECK:   integer :: a = 0
!CHECK:   !DEF: /block02/OtherConstruct1/BlockConstruct1/b SAVE ObjectEntity INTEGER(4)
!CHECK:   integer, save :: b
!CHECK:   !DEF: /block02/OtherConstruct1/BlockConstruct1/c (InDataStmt) ObjectEntity INTEGER(4)
!CHECK:   integer c
!CHECK:   !REF: /block02/OtherConstruct1/BlockConstruct1/c
!CHECK:   data c/0/
!CHECK:   !DEF: /block02/OtherConstruct1/BlockConstruct1/d ObjectEntity INTEGER(4)
!CHECK:   integer d
!CHECK:   !REF: /block02/OtherConstruct1/BlockConstruct1/a
!CHECK:   a = a+1
!CHECK:   !REF: /block02/OtherConstruct1/BlockConstruct1/b
!CHECK:   b = b+1
!CHECK:   !REF: /block02/OtherConstruct1/BlockConstruct1/c
!CHECK:   c = c+1
!CHECK:   !REF: /block02/OtherConstruct1/BlockConstruct1/d
!CHECK:   d = 1
!CHECK:  end block
!CHECK: !$omp end parallel
!CHECK: end subroutine

subroutine block02()
  !$omp parallel default(private)
  block
    integer :: a = 0          ! implied SAVE (initialization)
    integer, save :: b        ! explicit SAVE
    integer :: c
    data c /0/                ! implied SAVE (DATA)
    integer :: d              ! automatic
    a = a + 1
    b = b + 1
    c = c + 1
    d = 1
  end block
  !$omp end parallel
end subroutine

!CHECK-LABEL: !DEF: /block03 (Subroutine) Subprogram
!CHECK: subroutine block03
!CHECK: !$omp parallel default(private)
!CHECK:  block
!CHECK:   !DEF: /block03/OtherConstruct1/BlockConstruct1/x ObjectEntity INTEGER(4)
!CHECK:   integer x
!CHECK:   save
!CHECK:   !REF: /block03/OtherConstruct1/BlockConstruct1/x
!CHECK:   x = 1
!CHECK:  end block
!CHECK: !$omp end parallel
!CHECK: end subroutine

subroutine block03()
  !$omp parallel default(private)
  block
    integer :: x
    save                      ! bare SAVE scoped to the BLOCK
    x = 1
  end block
  !$omp end parallel
end subroutine

!CHECK-LABEL: !DEF: /block04 (Subroutine) Subprogram
!CHECK: subroutine block04
!CHECK:  block
!CHECK:   !DEF: /block04/BlockConstruct1/s ObjectEntity INTEGER(4)
!CHECK:   integer :: s = 1
!CHECK:   !DEF: /block04/BlockConstruct1/s2 SAVE ObjectEntity INTEGER(4)
!CHECK:   integer, save :: s2
!CHECK:   !DEF: /block04/BlockConstruct1/auto ObjectEntity INTEGER(4)
!CHECK:   integer auto
!CHECK:   !REF: /block04/BlockConstruct1/auto
!CHECK:   auto = 1
!CHECK: !$omp task
!CHECK:   !DEF: /block04/BlockConstruct1/OtherConstruct1/s (OmpShared) HostAssoc INTEGER(4)
!CHECK:   s = 2
!CHECK:   !DEF: /block04/BlockConstruct1/OtherConstruct1/s2 (OmpShared) HostAssoc INTEGER(4)
!CHECK:   s2 = 3
!CHECK:   !DEF: /block04/BlockConstruct1/OtherConstruct1/auto (OmpFirstPrivate, OmpImplicit) HostAssoc INTEGER(4)
!CHECK:   auto = 4
!CHECK: !$omp end task
!CHECK:  end block
!CHECK: end subroutine

subroutine block04()
  block
    integer :: s = 1
    integer, save :: s2
    integer :: auto
    auto = 1
    !$omp task
      s = 2
      s2 = 3
      auto = 4
    !$omp end task
  end block
end subroutine
