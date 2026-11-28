! RUN: %flang_fc1 -fopenmp -fopenmp-version=52 -emit-hlfir %s -o - | FileCheck %s

! A clause expression of an inner leaf construct sees the outer leaf constructs
! of the same compound directive.

module clause_context
contains
  pure integer function chunk(n)
    integer, intent(in) :: n
    !$omp declare variant(parallel_chunk) match(construct={parallel})
    !$omp declare variant(target_parallel_chunk) &
    !$omp& match(construct={target, parallel})
    !$omp declare variant(teams_parallel_chunk) &
    !$omp& match(construct={teams, parallel})
    chunk = n
  end function

  pure integer function parallel_chunk(n)
    integer, intent(in) :: n
    parallel_chunk = n
  end function

  pure integer function target_parallel_chunk(n)
    integer, intent(in) :: n
    target_parallel_chunk = n
  end function

  pure integer function teams_parallel_chunk(n)
    integer, intent(in) :: n
    teams_parallel_chunk = n
  end function

  ! CHECK-LABEL: func.func @_QMclause_contextPcombined(
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMclause_contextPparallel_chunk(
  ! CHECK: omp.wsloop
  subroutine combined(n, a)
    integer :: n, i, a(n)
    !$omp parallel do schedule(static, chunk(n))
    do i = 1, n
      a(i) = i
    end do
  end subroutine

  ! CHECK-LABEL: func.func @_QMclause_contextPtarget_combined(
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMclause_contextPtarget_parallel_chunk(
  ! CHECK: omp.wsloop
  subroutine target_combined(n, a)
    integer :: n, i, a(n)
    !$omp target parallel do schedule(static, chunk(n)) map(tofrom: a)
    do i = 1, n
      a(i) = i
    end do
  end subroutine

  ! In the composite construct, omp.parallel is created before the DO clauses.
  ! CHECK-LABEL: func.func @_QMclause_contextPcomposite(
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMclause_contextPteams_parallel_chunk(
  ! CHECK: omp.distribute
  subroutine composite(n, a)
    integer :: n, i, a(n)
    !$omp teams distribute parallel do schedule(static, chunk(n))
    do i = 1, n
      a(i) = i
    end do
  end subroutine
end module
