! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | \
! RUN:   FileCheck %s --implicit-check-not='fir.call @_QMloop_contextPparallel_count'
! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-fir %s -o - | \
! RUN:   FileCheck %s --implicit-check-not='fir.call @_QMloop_contextPparallel_count'

module loop_context
contains
  pure integer function teams_bound(n)
    integer, intent(in) :: n
    teams_bound = n
  end function

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
    !$omp declare variant(teams_bound) match(construct={teams})
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

  pure integer function cpu_count(n)
    integer, intent(in) :: n
    cpu_count = n
  end function

  pure integer function scored_count(n)
    integer, intent(in) :: n
    scored_count = n + 1
  end function

  pure integer function parallel_count(n)
    integer, intent(in) :: n
    parallel_count = n + 2
  end function

  pure logical function do_pred(n)
    integer, intent(in) :: n
    do_pred = n > 0
  end function

  pure logical function pred(n)
    integer, intent(in) :: n
    !$omp declare variant(do_pred) match(construct={do})
    pred = n > 0
  end function

  pure integer function thread_count(n)
    integer, intent(in) :: n
    !$omp declare variant(cpu_count) match(device={kind(cpu)})
    !$omp declare variant(parallel_count) &
    !$omp& match(construct={parallel}, device={kind(cpu)})
    !$omp declare variant(scored_count) &
    !$omp& match(implementation={vendor(score(3): llvm)})
    thread_count = n
  end function

  ! DIST_SCHEDULE sees only TEAMS. NUM_THREADS sees TEAMS, DISTRIBUTE, so
  ! CPU scores 5 and beats the vendor score of 4. A leaked PARALLEL would
  ! make parallel_count a strict superset of cpu_count. The bound sees PARALLEL.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite_clauses(
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite_clauses(n, a)
    integer :: n, a(n), i
    !$omp teams distribute parallel do dist_schedule(static, bound(n)) &
    !$omp& num_threads(thread_count(n))
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine

  ! The SIMD composite must evaluate its clauses in the same source contexts.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite_simd_clauses(
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPdo_pred(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite_simd_clauses(n, a)
    integer :: n, a(n), i
    !$omp teams distribute parallel do simd dist_schedule(static, bound(n)) &
    !$omp& num_threads(thread_count(n)) if(simd: pred(n))
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
  end subroutine

  ! SIMD clauses see DO, but the following loop bound and post-loop call do not.
  ! CHECK-LABEL: func.func @_QMloop_contextPdo_simd_clauses(
  ! CHECK: fir.call @_QMloop_contextPdo_pred(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPpred(
  ! CHECK: return
  subroutine do_simd_clauses(n, a, flag)
    integer :: n, a(n), i
    logical :: flag
    !$omp do simd if(simd: pred(n))
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
    flag = pred(n)
  end subroutine

  ! The explicit DO/SIMD nesting also exposes DO to the SIMD clause.
  ! CHECK-LABEL: func.func @_QMloop_contextPnested_simd_clauses(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_pred(
  ! CHECK: omp.simd
  ! CHECK: return
  subroutine nested_simd_clauses(n, a)
    integer :: n, a(n, n), i, j
    !$omp do
    do i = 1, n
      !$omp simd if(simd: pred(n))
      do j = 1, n
        a(i, j) = i + j
      end do
    end do
  end subroutine
end module
