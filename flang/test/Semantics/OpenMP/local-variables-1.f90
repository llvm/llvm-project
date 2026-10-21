!RUN: %flang_fc1 -fdebug-unparse-with-symbols -fopenmp %s | FileCheck %s

!Make sure that the local `bbb`s are their own entities.

!CHECK-LABEL:  !DEF: /f00 (Subroutine) Subprogram
!CHECK-NEXT:   subroutine f00
!CHECK-NEXT:    !DEF: /f00/i ObjectEntity INTEGER(4)
!CHECK-NEXT:    integer i
!CHECK-NEXT:   !$omp parallel
!CHECK-NEXT:    block
!CHECK-NEXT:     block
!CHECK-NEXT:      !DEF: /f00/OtherConstruct1/BlockConstruct1/BlockConstruct1/bbb ObjectEntity INTEGER(4)
!CHECK-NEXT:      integer bbb
!CHECK-NEXT:      !REF: /f00/OtherConstruct1/BlockConstruct1/BlockConstruct1/bbb
!CHECK-NEXT:      bbb = 1
!CHECK-NEXT:     end block
!CHECK-NEXT:     block
!CHECK-NEXT:      !DEF: /f00/OtherConstruct1/BlockConstruct1/BlockConstruct2/bbb ObjectEntity INTEGER(4)
!CHECK-NEXT:      integer bbb
!CHECK-NEXT:      !REF: /f00/OtherConstruct1/BlockConstruct1/BlockConstruct2/bbb
!CHECK-NEXT:      bbb = 2
!CHECK-NEXT:     end block
!CHECK-NEXT:    end block
!CHECK-NEXT:   !$omp end parallel
!CHECK-NEXT:   end subroutine

subroutine f00()
  integer :: i
  !$omp parallel
  block
    block
      integer :: bbb
      bbb = 1
    end block
    block
      integer :: bbb
      bbb = 2
    end block
  end block
  !$omp end parallel
end subroutine


!CHECK-LABEL:  !DEF: /f01 (Subroutine) Subprogram
!CHECK-NEXT:   subroutine f01
!CHECK-NEXT:    !DEF: /f01/i ObjectEntity INTEGER(4)
!CHECK-NEXT:    integer i
!CHECK-NEXT:    !DEF: /f01/bbb ObjectEntity INTEGER(4)
!CHECK-NEXT:    integer bbb
!CHECK-NEXT:    !REF: /f01/bbb
!CHECK-NEXT:    bbb = 0
!CHECK-NEXT:   !$omp parallel
!CHECK-NEXT:    block
!CHECK-NEXT:     !DEF: /f01/OtherConstruct1/bbb (OmpShared) HostAssoc INTEGER(4)
!CHECK-NEXT:     bbb = 1234
!CHECK-NEXT:     block
!CHECK-NEXT:      !DEF: /f01/OtherConstruct1/BlockConstruct1/BlockConstruct1/bbb ObjectEntity INTEGER(4)
!CHECK-NEXT:      integer bbb
!CHECK-NEXT:      !REF: /f01/OtherConstruct1/BlockConstruct1/BlockConstruct1/bbb
!CHECK-NEXT:      bbb = 1
!CHECK-NEXT:     end block
!CHECK-NEXT:     block
!CHECK-NEXT:      !DEF: /f01/OtherConstruct1/BlockConstruct1/BlockConstruct2/bbb ObjectEntity INTEGER(4)
!CHECK-NEXT:      integer bbb
!CHECK-NEXT:      !REF: /f01/OtherConstruct1/BlockConstruct1/BlockConstruct2/bbb
!CHECK-NEXT:      bbb = 2
!CHECK-NEXT:     end block
!CHECK-NEXT:    end block
!CHECK-NEXT:   !$omp end parallel
!CHECK-NEXT:    !REF: /f01/bbb
!CHECK-NEXT:    print *, bbb
!CHECK-NEXT:   end subroutine

subroutine f01()
  integer :: i
  integer :: bbb

  bbb = 0

  !$omp parallel
  block
    bbb = 1234
    block
      integer :: bbb
      bbb = 1
    end block
    block
      integer :: bbb
      bbb = 2
    end block
  end block
  !$omp end parallel

  print *, bbb
end subroutine
