! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 -o - %s | FileCheck %s

! Variant resolution replaces base procedure calls with variant calls
! inside matching OpenMP construct regions.

module m
contains
  subroutine variant_parallel()
  end subroutine
  subroutine base_parallel()
    !$omp declare variant(variant_parallel) match(construct={parallel})
  end subroutine
  subroutine variant_teams()
  end subroutine
  subroutine base_teams()
    !$omp declare variant(variant_teams) match(construct={teams})
  end subroutine
end module m

! CHECK-LABEL: func.func {{.*}}test_parallel
! CHECK:         omp.parallel {
! CHECK:           fir.call {{.*}}variant_parallel()
! CHECK-NOT:       fir.call {{.*}}base_parallel
! CHECK:           omp.terminator
! CHECK:         }
subroutine test_parallel()
  use m
  !$omp parallel
    call base_parallel()
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func {{.*}}test_teams
! CHECK:         omp.teams {
! CHECK:           fir.call {{.*}}variant_teams()
! CHECK-NOT:       fir.call {{.*}}base_teams
! CHECK:           omp.terminator
! CHECK:         }
subroutine test_teams()
  use m
  !$omp teams
    call base_teams()
  !$omp end teams
end subroutine

!Calls without omp directives

! CHECK-LABEL: func.func {{.*}}test_no_construct
! CHECK:         fir.call {{.*}}base_parallel()
! CHECK:         fir.call {{.*}}base_teams()
! CHECK-NOT:     fir.call {{.*}}variant
subroutine test_no_construct()
  use m
  call base_parallel()
  call base_teams()
end subroutine

! Test base:variant form with construct={parallel}
! CHECK-LABEL: func.func {{.*}}test_base_variant_form
! CHECK:         omp.parallel {
! CHECK:           fir.call {{.*}}test_base_variant_form{{.*}}variant_sub()
! CHECK:           omp.terminator
! CHECK:         }
subroutine test_base_variant_form()
  !$omp declare variant(base_sub:variant_sub) match(construct={parallel})
  !$omp parallel
    call base_sub()
  !$omp end parallel
contains
  subroutine variant_sub()
  end subroutine
  subroutine base_sub()
  end subroutine
end subroutine

! Test declare variant with user={condition(expr)} only.
! The base procedure body should contain a conditional branch:
! if condition is true, call variant and return; else execute base body.
module m_cond
  implicit none
  logical :: use_variant = .true.
contains
  subroutine foo_variant_cond()
  end subroutine
  subroutine foo_cond()
    !$omp declare variant(foo_variant_cond) match(user={condition(use_variant)})
  end subroutine
end module m_cond

! CHECK-LABEL: func.func {{.*}}foo_cond{{.*}}
! CHECK:         %[[COND:.*]] = fir.convert %{{.*}} : (!fir.logical<4>) -> i1
! CHECK:         cf.cond_br %[[COND]], ^[[THEN:.*]], ^[[ELSE:.*]]
! CHECK:       ^[[THEN]]:
! CHECK:         fir.call {{.*}}foo_variant_cond()
! CHECK:         return
! CHECK:       ^[[ELSE]]:
subroutine test_condition()
  use m_cond
  call foo_cond()
end subroutine

! Test declare variant with both construct={dispatch} and user={condition(expr)}.
! Inside !$omp dispatch, the call site should generate a conditional
! if/else selecting between variant and base calls.
module m_both
  implicit none
  logical :: use_variant_both = .true.
contains
  subroutine bar_variant()
  end subroutine
  subroutine bar()
    !$omp declare variant(bar_variant) &
    !$omp& match(construct={dispatch}, user={condition(use_variant_both)})
  end subroutine
end module m_both

! CHECK-LABEL: func.func {{.*}}test_dispatch_both
! CHECK:         fir.if %{{.*}} {
! CHECK:           fir.call {{.*}}bar_variant()
! CHECK:         } else {
! CHECK:           fir.call {{.*}}bar()
! CHECK:         }
subroutine test_dispatch_both()
  use m_both
  !$omp dispatch
  call bar()
  !$omp end dispatch
end subroutine

! Verify that without dispatch, the base call is preserved.
! CHECK-LABEL: func.func {{.*}}test_no_dispatch_both
! CHECK:         fir.call {{.*}}bar()
! CHECK-NOT:     fir.call {{.*}}bar_variant
subroutine test_no_dispatch_both()
  use m_both
  call bar()
end subroutine
