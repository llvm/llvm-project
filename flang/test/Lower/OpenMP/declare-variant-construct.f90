! RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 %s -o - | FileCheck %s

! DECLARE VARIANT callee resolution with combined/composite construct
! selectors.

! The combined directive selector decomposes to {target, teams}; it matches
! only when both constructs enclose the call.

! CHECK-LABEL: func.func @_QPtest_combined_match
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: fir.call @_QFbase_ttPvsub(){{.*}}: () -> ()
subroutine test_combined_match
  !$omp target
  !$omp teams
  call base_tt()
  !$omp end teams
  !$omp end target
end subroutine test_combined_match

subroutine base_tt
  !$omp declare variant (base_tt:vsub) match (construct={target teams})
contains
  subroutine vsub
  end subroutine
end subroutine base_tt

! Only `target` encloses the call, so the {target, teams} selector does not
! match and the base call is kept.

! CHECK-LABEL: func.func @_QPtest_combined_partial
! CHECK: omp.target
! CHECK: fir.call @_QPbase_tt2(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_tt2Pvsub
subroutine test_combined_partial
  !$omp target
  call base_tt2()
  !$omp end target
end subroutine test_combined_partial

subroutine base_tt2
  !$omp declare variant (base_tt2:vsub) match (construct={target teams})
contains
  subroutine vsub
  end subroutine
end subroutine base_tt2

! CHECK-LABEL: func.func @_QPtest_set_match
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase_tpPvsub(){{.*}}: () -> ()
subroutine test_set_match
  !$omp target
  !$omp parallel
  call base_tp()
  !$omp end parallel
  !$omp end target
end subroutine test_set_match

subroutine base_tp
  !$omp declare variant (base_tp:vsub) match (construct={target, parallel})
contains
  subroutine vsub
  end subroutine
end subroutine base_tp

! Inside `parallel` alone the {target, parallel} selector is not satisfied,
! so the base call is kept.

! CHECK-LABEL: func.func @_QPtest_set_no_match
! CHECK: omp.parallel
! CHECK: fir.call @_QPbase_tp2(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_tp2Pvsub
subroutine test_set_no_match
  !$omp parallel
  call base_tp2()
  !$omp end parallel
end subroutine test_set_no_match

subroutine base_tp2
  !$omp declare variant (base_tp2:vsub) match (construct={target, parallel})
contains
  subroutine vsub
  end subroutine
end subroutine base_tp2

! A `teams` construct between `target` and `parallel` does not prevent the
! match: {target, parallel} matches as an ordered subsequence of the enclosing
! target>teams>parallel context.

! CHECK-LABEL: func.func @_QPtest_set_match_nested_teams
! CHECK: omp.target
! CHECK: omp.teams
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase_tpPvsub(){{.*}}: () -> ()
subroutine test_set_match_nested_teams
  !$omp target
  !$omp teams
  !$omp parallel
  call base_tp()
  !$omp end parallel
  !$omp end teams
  !$omp end target
end subroutine test_set_match_nested_teams

subroutine base_rank
  !$omp declare variant (base_rank:vsub_par) match (construct={parallel})
  !$omp declare variant (base_rank:vsub_tp) match (construct={target, parallel})
contains
  subroutine vsub_par
  end subroutine
  subroutine vsub_tp
  end subroutine
end subroutine base_rank

! Inside target>parallel both variants apply; the more specific
! {target, parallel} selector outranks {parallel}.

! CHECK-LABEL: func.func @_QPtest_rank_specific
! CHECK: omp.target
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase_rankPvsub_tp(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_rankPvsub_par
subroutine test_rank_specific
  !$omp target
  !$omp parallel
  call base_rank()
  !$omp end parallel
  !$omp end target
end subroutine test_rank_specific

! Inside parallel alone only {parallel} applies.

! CHECK-LABEL: func.func @_QPtest_rank_parallel_only
! CHECK: omp.parallel
! CHECK: fir.call @_QFbase_rankPvsub_par(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_rankPvsub_tp
subroutine test_rank_parallel_only
  !$omp parallel
  call base_rank()
  !$omp end parallel
end subroutine test_rank_parallel_only

subroutine base_score
  !$omp declare variant (base_score:vsub_lo) match (user={condition(score(1): .true.)})
  !$omp declare variant (base_score:vsub_hi) match (user={condition(score(100): .true.)})
contains
  subroutine vsub_lo
  end subroutine
  subroutine vsub_hi
  end subroutine
end subroutine base_score

! Both conditions are statically true; the higher score wins.

! CHECK-LABEL: func.func @_QPtest_score_ranking
! CHECK: fir.call @_QFbase_scorePvsub_hi(){{.*}}: () -> ()
! CHECK-NOT: fir.call @_QFbase_scorePvsub_lo
subroutine test_score_ranking
  call base_score()
end subroutine test_score_ranking
