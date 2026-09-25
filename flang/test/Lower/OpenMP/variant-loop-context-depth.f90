! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s --implicit-check-not=omp.taskyield \
! RUN:   --implicit-check-not='fir.call @_QMloop_depthPdo_value'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-fir %s -o - | \
! RUN:   FileCheck %s --implicit-check-not=omp.taskyield \
! RUN:   --implicit-check-not='fir.call @_QMloop_depthPdo_value'

module loop_depth
contains
  integer function do_value()
    do_value = 1
  end function

  integer function scored_value()
    scored_value = 2
  end function

  integer function value()
    !$omp declare variant(do_value) match(construct={do})
    !$omp declare variant(scored_value) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    value = 0
  end function

  integer function requiring_do()
    !$omp declare variant(do_value) match(construct={do})
    requiring_do = 0
  end function

  ! At depth one CPU scores 3, below the user selector's score of 4.
  ! Counting the DO twice would select TASKYIELD instead.
  ! CHECK-LABEL: func.func @_QMloop_depthPbare_do(
  ! CHECK: omp.wsloop
  ! CHECK: omp.taskwait
  ! CHECK: return
  subroutine bare_do(n)
    integer :: n, i
    !$omp do
    do i = 1, n
      !$omp metadirective when(device={kind(cpu)}: taskyield) &
      !$omp& when(user={condition(score(3): .true.)}: taskwait)
    end do
  end subroutine

  ! DO at position two scores 3, below the vendor selector's score of 4.
  ! The same context must be used for both kinds of variant selection.
  ! CHECK-LABEL: func.func @_QMloop_depthPcombined(
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK: omp.taskwait
  ! CHECK: fir.call @_QMloop_depthPscored_value()
  ! CHECK: return
  subroutine combined(n, a)
    integer :: n, i, a(n)
    !$omp parallel do
    do i = 1, n
      !$omp metadirective when(construct={do}: taskyield) &
      !$omp& when(implementation={vendor(score(3): llvm)}: taskwait)
      a(i) = value()
    end do
  end subroutine

  ! Intervening code is outside the innermost DO evaluation, but shares the
  ! same OpenMP construct context as the collapsed loop body. The competing
  ! call rejects a duplicated DO; the DO-only call rejects a missing DO.
  ! CHECK-LABEL: func.func @_QMloop_depthPcollapsed(
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_depthPscored_value()
  ! CHECK: fir.call @_QMloop_depthPdo_value()
  ! CHECK: fir.call @_QMloop_depthPscored_value()
  ! CHECK: fir.call @_QMloop_depthPdo_value()
  ! CHECK: fir.call @_QMloop_depthPscored_value()
  ! CHECK: fir.call @_QMloop_depthPdo_value()
  ! CHECK: return
  subroutine collapsed(n, a)
    integer :: n, i, j, a(n, n)
    !$omp parallel do collapse(2)
    do i = 1, n
      a(i, 1) = value() + requiring_do()
      do j = 1, n
        a(i, j) = value() + requiring_do()
      end do
      a(i, n) = value() + requiring_do()
    end do
  end subroutine

  ! A nested TASK contributes a position, but must not move the enclosing DO.
  ! CHECK-LABEL: func.func @_QMloop_depthPnested_task(
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK: omp.task
  ! CHECK: omp.taskwait
  ! CHECK: return
  subroutine nested_task(n)
    integer :: n, i
    !$omp parallel do
    do i = 1, n
      !$omp task
        !$omp metadirective when(construct={do}: taskyield) &
        !$omp& when(implementation={vendor(score(3): llvm)}: taskwait)
      !$omp end task
    end do
  end subroutine

  ! TARGET hides the outer PARALLEL. TARGET, PARALLEL, DO give depth three:
  ! CPU scores 9 and loses to the user selector's score of 10.
  ! CHECK-LABEL: func.func @_QMloop_depthPtarget_depth(
  ! CHECK: omp.parallel
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: omp.wsloop
  ! CHECK: omp.taskwait
  ! CHECK: return
  subroutine target_depth(n)
    integer :: n, i
    !$omp parallel
      !$omp target
        !$omp parallel do
        do i = 1, n
          !$omp metadirective when(device={kind(cpu)}: taskyield) &
          !$omp& when(user={condition(score(9): .true.)}: taskwait)
        end do
      !$omp end target
    !$omp end parallel
  end subroutine
end module
