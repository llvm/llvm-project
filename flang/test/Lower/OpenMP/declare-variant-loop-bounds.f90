! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s

module loop_context
contains
  pure integer function parallel_bound(n)
    integer, intent(in) :: n
    parallel_bound = n
  end function

  pure integer function do_bound(n)
    integer, intent(in) :: n
    do_bound = n
  end function

  pure integer function bound(n)
    integer, intent(in) :: n
    !$omp declare variant(parallel_bound) match(construct={parallel})
    !$omp declare variant(do_bound) match(construct={do})
    bound = n
  end function

  ! NUM_THREADS is evaluated outside PARALLEL. The DO bound is inside
  ! PARALLEL but outside DO, and the loop body is inside both constructs.
  ! CHECK-LABEL: func.func @_QMloop_contextPcombined(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: omp.parallel
  ! CHECK-NOT: fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine combined(n, a)
    integer, intent(in) :: n
    integer :: a(n), i
    !$omp parallel do num_threads(bound(n))
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine

  ! Explicit nesting must select the same variants as the combined spelling.
  ! CHECK-LABEL: func.func @_QMloop_contextPexplicit_nest(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: omp.parallel
  ! CHECK-NOT: fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine explicit_nest(n, a)
    integer, intent(in) :: n
    integer :: a(n), i
    !$omp parallel num_threads(bound(n))
      !$omp do
      do i = 1, bound(n)
        a(i) = bound(n)
      end do
    !$omp end parallel
  end subroutine

  ! Composite lowering creates PARALLEL separately from the loop wrappers.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite(
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK-NOT: fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite(n, a)
    integer, intent(in) :: n
    integer :: a(n), i
    !$omp teams distribute parallel do
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine

  ! The SIMD composite follows a separate lowering path.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite_simd(
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK-NOT: fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite_simd(n, a)
    integer, intent(in) :: n
    integer :: a(n), i
    !$omp teams distribute parallel do simd
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine

  ! A standalone DO does not contribute its own context to its bounds.
  ! CHECK-LABEL: func.func @_QMloop_contextPstandalone_do(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine standalone_do(n, a)
    integer, intent(in) :: n
    integer :: a(n), i
    !$omp do
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine
end module
