! RUN: %flang_fc1 -fdebug-pre-fir-tree -fopenmp %s | FileCheck %s

! Test Pre-FIR Tree captures OpenMP related constructs

! CHECK: Program test_omp
program test_omp
  ! CHECK: PrintStmt
  print *, "sequential"

  ! CHECK: <<OpenMPConstruct>>
  !$omp parallel
    ! CHECK: PrintStmt
    print *, "in omp //"
    ! CHECK: <<OpenMPConstruct>>
    !$omp do
    ! CHECK: <<DoConstruct>>
    ! CHECK: LabelDoStmt
    do i=1,100
      ! CHECK: PrintStmt
      print *, "in omp do"
    ! CHECK: EndDoStmt
    end do
    ! CHECK: <<End DoConstruct>>
    ! CHECK: OmpEndLoopDirective
    !$omp end do
    ! CHECK: <<End OpenMPConstruct>>

    ! CHECK: PrintStmt
    print *, "not in omp do"

    ! CHECK: <<OpenMPConstruct>>
    !$omp do
    ! CHECK: <<DoConstruct>>
    ! CHECK: LabelDoStmt
    do i=1,100
      ! CHECK: PrintStmt
      print *, "in omp do"
    ! CHECK: EndDoStmt
    end do
    ! CHECK: <<End DoConstruct>>
    ! CHECK: <<End OpenMPConstruct>>
    ! CHECK-NOT: OmpEndLoopDirective
    ! CHECK: PrintStmt
    print *, "no in omp do"
  !$omp end parallel
    ! CHECK: <<End OpenMPConstruct>>

  ! CHECK: PrintStmt
  print *, "sequential again"

  ! CHECK: <<OpenMPConstruct>>
  !$omp task
    ! CHECK: PrintStmt
    print *, "in task"
  !$omp end task
  ! CHECK: <<End OpenMPConstruct>>

  ! CHECK: PrintStmt
  print *, "sequential again"
end program
