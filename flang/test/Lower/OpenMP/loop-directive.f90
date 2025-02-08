! This test checks lowering of OpenMP loop Directive.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: omp.declare_reduction @[[RED:add_reduction_i32]] : i32
! CHECK: omp.private {type = private} @[[DUMMY_PRIV:.*test_privateEdummy_private.*]] : i32
! CHECK: omp.private {type = private} @[[I_PRIV:.*test_no_clausesEi.*]] : i32

! CHECK-LABEL: func.func @_QPtest_no_clauses
subroutine test_no_clauses()
  integer :: i, j, dummy = 1

  ! CHECK: omp.simd private(@[[I_PRIV]] %{{.*}}#0 -> %[[ARG:.*]] : !fir.ref<i32>) {
  ! CHECK-NEXT:   omp.loop_nest (%[[IV:.*]]) : i32 = (%{{.*}}) to (%{{.*}}) {{.*}} {
  ! CHECK:          %[[ARG_DECL:.*]]:2 = hlfir.declare %[[ARG]]
  ! CHECK:          fir.store %[[IV]] to %[[ARG_DECL]]#1 : !fir.ref<i32>
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

  ! CHECK: omp.loop private(@{{.*}} %{{.*}}#0 -> %{{.*}} : !{{.*}}) reduction
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
