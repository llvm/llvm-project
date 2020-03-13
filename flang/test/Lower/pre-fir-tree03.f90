! RUN: %f18 -fdebug-pre-fir-tree -fparse-only -fopenmp %s | FileCheck %s

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
    ! CHECK: <<EndDoConstruct>>
    ! CHECK: OmpEndLoopDirective
    !$omp end do
    ! CHECK: <<EndOpenMPConstruct>>

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
    ! CHECK: <<EndDoConstruct>>
    ! CHECK: <<EndOpenMPConstruct>>
    ! CHECK-NOT: OmpEndLoopDirective
    ! CHECK: PrintStmt
    print *, "no in omp do"
  !$omp end parallel
    ! CHECK: <<EndOpenMPConstruct>>

  ! CHECK: PrintStmt
  print *, "sequential again"

  ! CHECK: <<OpenMPConstruct>>
  !$omp task
    ! CHECK: PrintStmt
    print *, "in task"
  !$omp end task
  ! CHECK: <<EndOpenMPConstruct>>

  ! CHECK: PrintStmt
  print *, "sequential again"
end program
