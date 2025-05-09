! This test checks lowering of OpenMP loop Directive.

! REQUIRES: openmp_runtime

! RUN: %flang_fc1 -emit-hlfir %openmp_flags -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: omp.declare_reduction @[[RED:add_reduction_i32]] : i32
! CHECK: omp.private {type = private} @[[DUMMY_PRIV:.*test_privateEdummy_private.*]] : i32
! CHECK: omp.private {type = private} @[[I_PRIV:.*test_no_clausesEi.*]] : i32

! CHECK-LABEL: func.func @_QPtest_no_clauses
subroutine test_no_clauses()
  integer :: i, j, dummy = 1

  ! CHECK: omp.simd private(@[[I_PRIV]] %{{.*}}#0 -> %[[ARG:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:   omp.loop_nest (%[[IV:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) {{.*}} {
  ! CHECK:          %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG]]
  ! CHECK:          hlfir.assign %[[IV]] to %[[ARG_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:        }
  ! CHECK: }
  !$omp loop
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_collapse
subroutine test_collapse()
  integer :: i, j, dummy = 1
  ! CHECK: omp.simd private(@{{.*}} %{{.*}}#0 -> %{{.*}}, @{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
  ! CHECK-NEXT:   omp.loop_nest (%{{.*}}, %{{.*}}) : i32 {{.*}} {
  ! CHECK:        }
  ! CHECK: }
  !$omp loop collapse(2)
  do i=1,10
    do j=2,20
     dummy = dummy + 1
    end do
  end do
  !$omp end loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_private
subroutine test_private()
  integer :: i, dummy = 1
  ! CHECK: omp.simd private(@[[DUMMY_PRIV]] %{{.*}}#0 -> %[[DUMMY_ARG:.*]], @{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
  ! CHECK-NEXT:   omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) {{.*}} {
  ! CHECK:          %[[DUMMY_DECL:.*]]:2 = hlfir.declare %[[DUMMY_ARG]] {uniq_name = "_QFtest_privateEdummy"}
  ! CHECK:          %{{.*}} = fir.load %[[DUMMY_DECL]]#0
  ! CHECK:          hlfir.assign %{{.*}} to %[[DUMMY_DECL]]#0
  ! CHECK:        }
  ! CHECK: }
  !$omp loop private(dummy)
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine


! CHECK-LABEL: func.func @_QPtest_order
subroutine test_order()
  integer :: i, dummy = 1
  ! CHECK: omp.loop order(reproducible:concurrent) private(@{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
  ! CHECK: }
  !$omp loop order(concurrent)
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_reduction
subroutine test_reduction()
  integer :: i, dummy = 1

  ! CHECK: omp.simd private(@{{.*}} %{{.*}}#0 -> %{{.*}} : !{{.*}}) reduction
  ! CHECK-SAME:  (@[[RED]] %{{.*}}#0 -> %[[DUMMY_ARG:.*]] : !{{.*}}) {
  ! CHECK-NEXT:   omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) {{.*}} {
  ! CHECK:          %[[DUMMY_DECL:.*]]:2 = hlfir.declare %[[DUMMY_ARG]] {uniq_name = "_QFtest_reductionEdummy"}
  ! CHECK:          %{{.*}} = fir.load %[[DUMMY_DECL]]#0
  ! CHECK:          hlfir.assign %{{.*}} to %[[DUMMY_DECL]]#0
  ! CHECK:        }
  ! CHECK: }
  !$omp loop reduction(+:dummy)
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_bind
subroutine test_bind()
  integer :: i, dummy = 1
  ! CHECK: omp.simd private(@{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
  ! CHECK: }
  !$omp loop bind(thread)
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine

! CHECK-LABEL: func.func @_QPtest_nested_directives
subroutine test_nested_directives
  implicit none
  integer, parameter :: N = 100000
  integer a(N), b(N), c(N)
  integer j,i, num, flag;
  num = N

  ! CHECK: omp.teams {

  ! Verify the first `loop` directive was combined with `target teams` into 
  ! `target teams distribute parallel do`.
  ! CHECK:   omp.parallel {{.*}} {
  ! CHECK:     omp.distribute {
  ! CHECK:       omp.wsloop {
  ! CHECK:         omp.loop_nest {{.*}} {

  ! Very the second `loop` directive was rewritten to `simd`.
  ! CHECK:           omp.simd {{.*}} {
  ! CHECK:             omp.loop_nest {{.*}} {
  ! CHECK:             }
  ! CHECK:           }

  ! CHECK:         }
  ! CHECK:       } {omp.composite}
  ! CHECK:     } {omp.composite}
  ! CHECK:   } {omp.composite}
  ! CHECK: }
  !$omp target teams map(to: a,b) map(from: c)
  !$omp loop
  do j=1,1000
    !$omp loop
    do i=1,N
      c(i) = a(i) * b(i)
    end do
  end do
  !$omp end target teams
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_bind_teams
subroutine test_standalone_bind_teams
  implicit none
  integer, parameter :: N = 100000
  integer a(N), b(N), c(N)
  integer j,i, num, flag;
  num = N

  ! CHECK:     omp.distribute
  ! CHECK-SAME:  private(@{{.*}}Ea_private_box_100000xi32 {{[^,]*}},
  ! CHECK-SAME:          @{{.*}}Ei_private_i32 {{.*}} : {{.*}}) {
  ! CHECK:       omp.loop_nest {{.*}} {
  ! CHECK:       }
  ! CHECK:     }
  !$omp loop bind(teams) private(a)
  do i=1,N
    c(i) = a(i) * b(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_bind_parallel
subroutine test_standalone_bind_parallel
  implicit none
  integer, parameter :: N = 100000
  integer a(N), b(N), c(N)
  integer j,i, num, flag;
  num = N

  ! CHECK:     omp.wsloop
  ! CHECK-SAME:  private(@{{.*}}Ea_private_box_100000xi32 {{[^,]*}},
  ! CHECK-SAME:          @{{.*}}Ei_private_i32 {{.*}} : {{.*}}) {
  ! CHECK:       omp.loop_nest {{.*}} {
  ! CHECK:       }
  ! CHECK:     }
  !$omp loop bind(parallel) private(a)
  do i=1,N
    c(i) = a(i) * b(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPteams_loop_cannot_be_parallel_for
subroutine teams_loop_cannot_be_parallel_for
  implicit none
  integer :: iter, iter2, val(20)
  val = 0
  ! CHECK: omp.teams {

  ! Verify the outer `loop` directive was mapped to only `distribute`.
  ! CHECK-NOT: omp.parallel {{.*}}
  ! CHECK:     omp.distribute {{.*}} {
  ! CHECK-NEXT:  omp.loop_nest {{.*}} {

  ! Verify the inner `loop` directive was mapped to a worksharing loop.
  ! CHECK:         omp.wsloop {{.*}} {
  ! CHECK-NEXT:      omp.loop_nest {{.*}} {
  ! CHECK:           }
  ! CHECK:         }

  ! CHECK:       }
  ! CHECK:     }

  ! CHECK: }
  !$omp target teams loop map(tofrom:val)
  DO iter = 1, 5
    !$omp loop bind(parallel)
    DO iter2 = 1, 5
      val(iter+iter2) = iter+iter2
    END DO
  END DO
end subroutine

subroutine foo()
end subroutine

! CHECK-LABEL: func.func @_QPteams_loop_cannot_be_parallel_for_2
subroutine teams_loop_cannot_be_parallel_for_2
  implicit none
  integer :: iter, val(20)
  val = 0

  ! CHECK: omp.teams {

  ! Verify the `loop` directive was mapped to only `distribute`.
  ! CHECK-NOT: omp.parallel {{.*}}
  ! CHECK:     omp.distribute {{.*}} {
  ! CHECK-NEXT:  omp.loop_nest {{.*}} {
  ! CHECK:         fir.call @_QPfoo
  ! CHECK:       }
  ! CHECK:     }

  ! CHECK: }
  !$omp target teams loop map(tofrom:val)
  DO iter = 1, 5
    call foo()
  END DO
end subroutine

! CHECK-LABEL: func.func @_QPteams_loop_can_be_parallel_for
subroutine teams_loop_can_be_parallel_for
  use omp_lib
  implicit none
  integer :: iter, tid, val(20)
  val = 0

  !CHECK: omp.teams {
  !CHECK:   omp.parallel {{.*}} {
  !CHECK:     omp.distribute {
  !CHECK:       omp.wsloop {
  !CHECK:         omp.loop_nest {{.*}} {
  !CHECK:           fir.call @omp_get_thread_num()
  !CHECK:         }
  !CHECK:       }
  !CHECK:     }
  !CHECK:   }
  !CHECK: }
  !$omp target teams loop map(tofrom:val)
  DO iter = 1, 5
    tid = omp_get_thread_num()
  END DO
end subroutine

! CHECK-LABEL: func.func @_QPteams_loop_cannot_be_parallel_for_4
subroutine teams_loop_cannot_be_parallel_for_4
  implicit none
  integer :: iter, iter2, tid, val(20)

  ! CHECK: omp.teams {

  ! Verify the outer `loop` directive was mapped to only `distribute`.
  ! CHECK-NOT: omp.parallel {{.*}}
  ! CHECK:     omp.distribute {{.*}} {
  ! CHECK-NEXT:  omp.loop_nest {{.*}} {

  ! Verify the inner `loop` directive was mapped to a worksharing loop.
  ! CHECK:         omp.wsloop {{.*}} {
  ! CHECK:           omp.loop_nest {{.*}} {
  ! CHECK:           }
  ! CHECK:         }

  ! CHECK:       }
  ! CHECK:     }

  ! CHECK: }
  !$omp target teams loop map(tofrom:val)
  DO iter = 1, 5
    !$omp parallel
    !$omp loop
    DO iter2 = 1, 5
      val(iter+iter2) = iter+iter2
    END DO
    !$omp end parallel
  END DO
end subroutine

! CHECK-LABEL: func.func @_QPloop_parallel_bind_reduction
subroutine loop_parallel_bind_reduction
  implicit none
  integer :: x, i

  ! CHECK: omp.wsloop
  ! CHECK-SAME: private(@{{[^[:space:]]+}} %{{[^[:space:]]+}}#0 -> %[[PRIV_ARG:[^[:space:]]+]] : !fir.ref<i32>)
  ! CHECK-SAME: reduction(@add_reduction_i32 %{{.*}}#0 -> %[[RED_ARG:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT: omp.loop_nest {{.*}} {
  ! CHECK-NEXT:   hlfir.declare %[[PRIV_ARG]] {uniq_name = "_QF{{.*}}Ei"}
  ! CHECK-NEXT:   hlfir.declare %[[RED_ARG]] {uniq_name = "_QF{{.*}}Ex"}
  ! CHECK:      }
  ! CHECK: }
  !$omp loop bind(parallel) reduction(+: x)
  do i = 0, 10
    x = x + i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPloop_teams_loop_reduction
subroutine loop_teams_loop_reduction
  implicit none
  integer :: x, i
  ! CHECK: omp.teams reduction(@add_reduction_i32 %{{.*}}#0 -> %[[TEAMS_RED_ARG:.*]] : !fir.ref<i32>) {
  ! CHECK:   omp.parallel
  ! CHECK-SAME: private(@{{[^[:space:]]+}} %{{[^[:space:]]+}}#0 -> %[[PRIV_ARG:[^[:space:]]+]] : !fir.ref<i32>) {
  ! CHECK:      omp.distribute {
  ! CHECK:        omp.wsloop
  ! CHECK-SAME:     reduction(@add_reduction_i32 %[[TEAMS_RED_ARG]] -> %[[RED_ARG:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:     omp.loop_nest {{.*}} {
  ! CHECK-NEXT:       hlfir.declare %[[PRIV_ARG]] {uniq_name = "_QF{{.*}}Ei"}
  ! CHECK-NEXT:       hlfir.declare %[[RED_ARG]] {uniq_name = "_QF{{.*}}Ex"}
  ! CHECK:          }
  ! CHECK:        }
  ! CHECK:      }
  ! CHECK:    }
  ! CHECK: }
  !$omp teams loop reduction(+: x)
  do i = 0, 10
    x = x + i
  end do
end subroutine


! Tests a regression when the pass encounters a multi-block `teams` region.
subroutine multi_block_teams
  implicit none
  integer :: i

  ! CHECK: omp.target {{.*}} {
  ! CHECK:   omp.teams {
  ! CHECK:   ^bb1:
  ! CHECK:     cf.br ^bb2
  ! CHECK:   ^bb2:
  ! CHECK:     omp.terminator
  ! CHECK:   }
  ! CHECK: }
  !$omp target teams
  select case (i)
  case(1)
  end select
  !$omp end target teams
end subroutine


! Verifies that reductions are hoisted to the parent `teams` directive and removed
! from the `loop` directive when `loop` is mapped to `distribute`.

! CHECK-LABEL: func.func @_QPteams_loop_cannot_be_parallel_for_with_reductions
subroutine teams_loop_cannot_be_parallel_for_with_reductions
  implicit none
  integer :: x, y, i, p

  ! CHECK: %[[ADD_RED:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QF{{.*}}Ex"}
  ! CHECK: %[[MUL_RED:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "_QF{{.*}}Ey"}
  ! CHECK: omp.teams reduction(
  ! CHECK-SAME:  @add_reduction_i32 %[[ADD_RED]]#0 -> %[[ADD_RED_ARG:[^[:space:]]*]], 
  ! CHECK-SAME:  @multiply_reduction_i32 %[[MUL_RED]]#0 -> %[[MUL_RED_ARG:.*]] : {{.*}}) {

  ! CHECK:       omp.distribute private(@{{.*}} %{{.*}} -> %{{.*}}, @{{.*}} %{{.*}} -> %{{.*}} : {{.*}}) {
  ! CHECK:         %[[ADD_RED_DECL:.*]]:2 = hlfir.declare %[[ADD_RED_ARG]] {uniq_name = "_QF{{.*}}Ex"}
  ! CHECK:         %[[MUL_RED_DECL:.*]]:2 = hlfir.declare %[[MUL_RED_ARG]] {uniq_name = "_QF{{.*}}Ey"}

  ! CHECK:         %[[ADD_RES:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
  ! CHECK:         hlfir.assign %[[ADD_RES]] to %[[ADD_RED_DECL]]#0 : i32, !fir.ref<i32>

  ! CHECK:         %[[MUL_RES:.*]] = arith.muli %{{.*}}, %{{.*}} : i32
  ! CHECK:         hlfir.assign %[[MUL_RES]] to %[[MUL_RED_DECL]]#0 : i32, !fir.ref<i32>
  ! CHECK:         omp.yield
  ! CHECK:       }
  ! CHECK:       omp.terminator
  ! CHECK: }
  !$omp teams loop reduction(+: x) reduction(*: y) private(p)
  do i = 1, 5
    call foo()
    x = x + i
    y = y * i
    p = 42
  end do
end subroutine
