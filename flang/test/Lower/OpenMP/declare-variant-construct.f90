! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! DECLARE VARIANT callee resolution with combined/composite construct
! selectors. The bases and their variants are sibling module procedures, so
! each variant is accessible at every reference to its base.

module m
contains
  subroutine base_tt
    !$omp declare variant (base_tt:vsub_tt) match (construct={target teams})
  end subroutine base_tt
  subroutine vsub_tt
  end subroutine vsub_tt

  subroutine base_tt2
    !$omp declare variant (base_tt2:vsub_tt2) match (construct={target teams})
  end subroutine base_tt2
  subroutine vsub_tt2
  end subroutine vsub_tt2

  subroutine base_tp
    !$omp declare variant (base_tp:vsub_tp) match (construct={target, parallel})
  end subroutine base_tp
  subroutine vsub_tp
  end subroutine vsub_tp

  subroutine base_tp2
    !$omp declare variant (base_tp2:vsub_tp2) match (construct={target, parallel})
  end subroutine base_tp2
  subroutine vsub_tp2
  end subroutine vsub_tp2

  subroutine base_rank
    !$omp declare variant (base_rank:vsub_par) match (construct={parallel})
    !$omp declare variant (base_rank:vsub_rank_tp) match (construct={target, parallel})
  end subroutine base_rank
  subroutine vsub_par
  end subroutine vsub_par
  subroutine vsub_rank_tp
  end subroutine vsub_rank_tp

  subroutine base_score
    !$omp declare variant (base_score:vsub_lo) match (user={condition(score(1): .true.)})
    !$omp declare variant (base_score:vsub_hi) match (user={condition(score(100): .true.)})
  end subroutine base_score
  subroutine vsub_lo
  end subroutine vsub_lo
  subroutine vsub_hi
  end subroutine vsub_hi

  ! The combined directive selector decomposes to {target, teams}; it matches
  ! only when both constructs enclose the call.

  ! CHECK-LABEL: func.func @_QMmPtest_combined_match
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: fir.call @_QMmPvsub_tt(){{.*}}: () -> ()
  subroutine test_combined_match
    !$omp target
    !$omp teams
    call base_tt()
    !$omp end teams
    !$omp end target
  end subroutine test_combined_match

  ! Only `target` encloses the call, so the {target, teams} selector does not
  ! match and the base call is kept.

  ! CHECK-LABEL: func.func @_QMmPtest_combined_partial
  ! CHECK: omp.target
  ! CHECK: fir.call @_QMmPbase_tt2(){{.*}}: () -> ()
  ! CHECK-NOT: fir.call @_QMmPvsub_tt2
  subroutine test_combined_partial
    !$omp target
    call base_tt2()
    !$omp end target
  end subroutine test_combined_partial

  ! CHECK-LABEL: func.func @_QMmPtest_set_match
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMmPvsub_tp(){{.*}}: () -> ()
  subroutine test_set_match
    !$omp target
    !$omp parallel
    call base_tp()
    !$omp end parallel
    !$omp end target
  end subroutine test_set_match

  ! Inside `parallel` alone the {target, parallel} selector is not satisfied,
  ! so the base call is kept.

  ! CHECK-LABEL: func.func @_QMmPtest_set_no_match
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMmPbase_tp2(){{.*}}: () -> ()
  ! CHECK-NOT: fir.call @_QMmPvsub_tp2
  subroutine test_set_no_match
    !$omp parallel
    call base_tp2()
    !$omp end parallel
  end subroutine test_set_no_match

  ! A `teams` construct between `target` and `parallel` does not prevent the
  ! match: {target, parallel} matches as an ordered subsequence of the enclosing
  ! target>teams>parallel context.

  ! CHECK-LABEL: func.func @_QMmPtest_set_match_nested_teams
  ! CHECK: omp.target
  ! CHECK: omp.teams
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMmPvsub_tp(){{.*}}: () -> ()
  subroutine test_set_match_nested_teams
    !$omp target
    !$omp teams
    !$omp parallel
    call base_tp()
    !$omp end parallel
    !$omp end teams
    !$omp end target
  end subroutine test_set_match_nested_teams

  ! Inside target>parallel both variants apply; the more specific
  ! {target, parallel} selector outranks {parallel}.

  ! CHECK-LABEL: func.func @_QMmPtest_rank_specific
  ! CHECK: omp.target
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMmPvsub_rank_tp(){{.*}}: () -> ()
  ! CHECK-NOT: fir.call @_QMmPvsub_par
  subroutine test_rank_specific
    !$omp target
    !$omp parallel
    call base_rank()
    !$omp end parallel
    !$omp end target
  end subroutine test_rank_specific

  ! Inside parallel alone only {parallel} applies.

  ! CHECK-LABEL: func.func @_QMmPtest_rank_parallel_only
  ! CHECK: omp.parallel
  ! CHECK: fir.call @_QMmPvsub_par(){{.*}}: () -> ()
  ! CHECK-NOT: fir.call @_QMmPvsub_rank_tp
  subroutine test_rank_parallel_only
    !$omp parallel
    call base_rank()
    !$omp end parallel
  end subroutine test_rank_parallel_only

  ! Both conditions are statically true; the higher score wins.

  ! CHECK-LABEL: func.func @_QMmPtest_score_ranking
  ! CHECK: fir.call @_QMmPvsub_hi(){{.*}}: () -> ()
  ! CHECK-NOT: fir.call @_QMmPvsub_lo
  subroutine test_score_ranking
    call base_score()
  end subroutine test_score_ranking
end module m
