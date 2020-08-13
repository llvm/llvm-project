! RUN: %f18 -fdebug-pre-fir-tree -fparse-only -fopenacc %s | FileCheck %s

! Test structure of the Pre-FIR tree with OpenACC construct

! CHECK: Subroutine foo
subroutine foo()
  ! CHECK-NEXT: <<OpenACCConstruct>>
  ! CHECK-NEXT: <<OpenACCConstruct>>
  !$acc parallel
  !$acc loop
  ! CHECK-NEXT: <<DoConstruct>>
  ! CHECK-NEXT: NonLabelDoStmt
  do i=1,5
    ! CHECK-NEXT: PrintStmt
    print *, "hey"
    ! CHECK-NEXT: <<DoConstruct>>
    ! CHECK-NEXT: NonLabelDoStmt
    do j=1,5
      ! CHECK-NEXT: PrintStmt
      print *, "hello", i, j
    ! CHECK-NEXT: EndDoStmt
    ! CHECK-NEXT: <<End DoConstruct>>
    end do
  ! CHECK-NEXT: EndDoStmt
  ! CHECK-NEXT: <<End DoConstruct>>
  end do
  !$acc end parallel
  ! CHECK-NEXT: <<End OpenACCConstruct>>
  ! CHECK-NEXT: <<End OpenACCConstruct>>
  ! CHECK-NEXT: ContinueStmt
end subroutine
! CHECK-NEXT: EndSubroutine foo

! CHECK: Subroutine foo
subroutine foo2()
  ! CHECK-NEXT: <<OpenACCConstruct>>
  !$acc parallel loop
  ! CHECK-NEXT: <<DoConstruct>>
  ! CHECK-NEXT: NonLabelDoStmt
  do i=1,5
  ! CHECK-NEXT: EndDoStmt
  ! CHECK-NEXT: <<End DoConstruct>>
  end do
  !$acc end parallel loop
  ! CHECK-NEXT: <<End OpenACCConstruct>>
  ! CHECK-NEXT: ContinueStmt
end subroutine
! CHECK-NEXT: EndSubroutine foo2

