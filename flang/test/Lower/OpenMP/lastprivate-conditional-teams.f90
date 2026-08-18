! Test lowering of `lastprivate(conditional:)` on a worksharing loop nested in a
! host `teams` region (teams is valid on the host without `target`).  The
! conditional-lastprivate reduction struct is a per-team (pinned) stack
! allocation inside the teams region -- placed before the enclosing
! omp.parallel -- so multiple teams each get their own copy.  The guarded
! copy-back runs in an omp.single sibling of the wsloop.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_teams_conditional_lp(n, x)
  implicit none
  integer, intent(in) :: n
  integer :: x
  integer :: i

  !$omp teams num_teams(4)
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 3) == 0) x = i
  end do
  !$omp end parallel do
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPtest_teams_conditional_lp
! CHECK:         omp.teams
! The reduction struct is a per-team (pinned) stack allocation inside the teams
! region, initialized before the enclosing parallel.
! CHECK:           %[[STRUCT:.*]] = fir.alloca {{.*}}_lp_cond_t{{.*}} {pinned}
! CHECK:           fir.coordinate_of %[[STRUCT]], x
! CHECK:           fir.coordinate_of %[[STRUCT]], $x
! CHECK:           fir.store %{{.*}} to %{{.*}} : !fir.ref<i64>
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! Guarded copy-back in an omp.single sibling of the wsloop.
! CHECK:             omp.single {
! CHECK:               fir.coordinate_of %[[STRUCT]], x
! CHECK:               fir.coordinate_of %[[STRUCT]], $x
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               fir.if
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }

! Host teams enclosing parallel sections with two conditional list items.
subroutine test_teams_sections_conditional_lp(x, y)
  implicit none
  integer :: x, y

  !$omp teams num_teams(4)
  !$omp parallel sections lastprivate(conditional: x, y)
  !$omp section
    x = 11
  !$omp section
    y = 22
  !$omp end parallel sections
  !$omp end teams
end subroutine

! CHECK-LABEL: func.func @_QPtest_teams_sections_conditional_lp
! CHECK:         omp.teams
! CHECK:           omp.parallel {
! CHECK:             omp.sections
! CHECK-SAME:          reduction(byref @lp_cond_byref_rec__lp_cond_t
! CHECK:             omp.single {
! Two guarded copy-backs, one per conditional list item.
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               arith.cmpi sge, %{{.*}}, %{{.*}} : i64
! CHECK:               omp.terminator
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           omp.terminator
! CHECK:         }
