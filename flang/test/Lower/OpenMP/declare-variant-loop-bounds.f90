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

  ! Original bounds are evaluated in the context before DISTRIBUTE.
  ! Emitting the composite PARALLEL region first must not affect selection.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite(
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK-NOT: fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
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
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
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
  ! make parallel_count a strict superset of cpu_count. Bounds see only TEAMS.
  ! CHECK-LABEL: func.func @_QMloop_contextPcomposite_clauses(
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite_clauses(n, a)
    integer :: n, a(n), i
    !$omp teams distribute parallel do dist_schedule(static, bound(n)) &
    !$omp& num_threads(thread_count(n))
    do i = bound(1), bound(n), bound(1)
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
  ! CHECK: fir.call @_QMloop_contextPscored_count(
  ! CHECK: omp.distribute
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine composite_simd_clauses(n, a)
    integer :: n, a(n), i
    !$omp teams distribute parallel do simd dist_schedule(static, bound(n)) &
    !$omp& num_threads(thread_count(n)) if(simd: pred(n))
    do i = 1, thread_count(n)
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
  ! SIMD clauses see TEAMS and DISTRIBUTE (CPU scores 5). The bound and
  ! post-loop call see only TEAMS (CPU scores 3, losing to vendor score 4).
  ! CHECK-LABEL: func.func @_QMloop_contextPdistribute_simd_clauses(
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: fir.call @_QMloop_contextPscored_count(
  ! CHECK: omp.distribute
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPscored_count(
  ! CHECK: return
  subroutine distribute_simd_clauses(n, a)
    integer :: n, a(n), i
    !$omp teams
    !$omp distribute simd if(simd: thread_count(n) > 0)
    do i = 1, thread_count(n)
      a(i) = i
    end do
    i = thread_count(n)
    !$omp end teams
  end subroutine

  ! CHECK-LABEL: func.func @_QMloop_contextPnested_distribute_simd_clauses(
  ! CHECK: omp.teams
  ! CHECK: omp.distribute
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: omp.simd
  ! CHECK: return
  subroutine nested_distribute_simd_clauses(n, a)
    integer :: n, a(n, n), i, j
    !$omp teams
    !$omp distribute
    do i = 1, n
      !$omp simd if(simd: thread_count(n) > 0)
      do j = 1, n
        a(i, j) = i + j
      end do
    end do
    !$omp end teams
  end subroutine

  ! The same original-bound rule applies without PARALLEL in the composite.
  ! CHECK-LABEL: func.func @_QMloop_contextPteams_distribute(
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMloop_contextPteams_bound(
  ! CHECK: fir.call @_QMloop_contextPscored_count(
  ! CHECK: omp.distribute
  ! CHECK: fir.call @_QMloop_contextPcpu_count(
  ! CHECK: fir.call @_QMloop_contextPscored_count(
  ! CHECK: return
  subroutine teams_distribute(n, a)
    integer :: n, a(n), i
    !$omp teams
    !$omp distribute
    do i = bound(1), thread_count(n)
      a(i) = thread_count(n)
    end do
    i = thread_count(n)
    !$omp end teams
  end subroutine

  ! Retain PARALLEL preceding DO in the same combined directive.
  ! CHECK-LABEL: func.func @_QMloop_contextPparallel_do_simd(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPdo_pred(
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.wsloop
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: return
  subroutine parallel_do_simd(n, a)
    integer :: n, a(n), i
    !$omp parallel do simd if(simd: pred(n))
    do i = 1, bound(n)
      a(i) = bound(n)
    end do
    i = bound(n)
  end subroutine

  ! An enclosing DO belongs to a different source loop and must remain active
  ! for the inner SIMD bounds. Only the inner directive's frames are filtered.
  ! CHECK-LABEL: func.func @_QMloop_contextPenclosing_do(
  ! CHECK: omp.wsloop
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: omp.simd
  ! CHECK: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: return
  subroutine enclosing_do(n, a)
    integer :: n, a(n,n), i, j
    !$omp do
    do i = 1, n
      !$omp simd
      do j = 1, bound(n)
        a(i,j) = bound(n)
      end do
    end do
    i = bound(n)
  end subroutine
  ! Lastprivate re-evaluates the bound inside WSLOOP for the copy-back test.
  ! Both evaluations belong to the original loop control, outside DO.
  ! CHECK-LABEL: func.func @_QMloop_contextPlastprivate_bound(
  ! CHECK: fir.call @_QMloop_contextPbound(
  ! CHECK: omp.wsloop
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: %[[COPY_UB:.*]] = fir.call @_QMloop_contextPbound(
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: arith.cmpi sgt, %{{[^,]+}}, %[[COPY_UB]]
  ! CHECK: fir.if
  ! CHECK: {{hlfir.assign|fir.store}}
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine lastprivate_bound(n, x)
    integer :: n, x, i
    !$omp do lastprivate(x)
    do i = 1, bound(n)
      x = i
    end do
  end subroutine

  ! Keep the entered PARALLEL constituent when re-evaluating the bound.
  ! CHECK-LABEL: func.func @_QMloop_contextPlastprivate_combined(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.wsloop
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: %[[PAR_UB:.*]] = fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: arith.cmpi sgt, %{{[^,]+}}, %[[PAR_UB]]
  ! CHECK: fir.if
  ! CHECK: {{hlfir.assign|fir.store}}
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine lastprivate_combined(n, x)
    integer :: n, x, i
    !$omp parallel do lastprivate(x)
    do i = 1, bound(n)
      x = i
    end do
  end subroutine

  ! A separately enclosing PARALLEL must also remain in the bound context.
  ! CHECK-LABEL: func.func @_QMloop_contextPlastprivate_nested(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: omp.wsloop
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: %[[NESTED_UB:.*]] = fir.call @_QMloop_contextPparallel_bound(
  ! CHECK: arith.cmpi sgt, %{{[^,]+}}, %[[NESTED_UB]]
  ! CHECK: fir.if
  ! CHECK: {{hlfir.assign|fir.store}}
  ! CHECK-NOT: fir.call @_QMloop_contextPdo_bound(
  ! CHECK: return
  subroutine lastprivate_nested(n, x)
    integer :: n, x, i
    !$omp parallel
    !$omp do lastprivate(x)
    do i = 1, bound(n)
      x = i
    end do
    !$omp end parallel
  end subroutine

end module
