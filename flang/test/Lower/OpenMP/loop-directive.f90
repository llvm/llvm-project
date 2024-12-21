! This test checks lowering of OpenMP loop Directive.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK: omp.declare_reduction @[[RED:add_reduction_i32]] : i32
! CHECK: omp.private {type = private} @[[DUMMY_PRIV:.*test_privateEdummy_private.*]] : !fir.ref<i32>
! CHECK: omp.private {type = private} @[[I_PRIV:.*test_no_clausesEi.*]] : !fir.ref<i32>

! CHECK-LABEL: func.func @_QPtest_no_clauses
subroutine test_no_clauses()
  integer :: i, j, dummy = 1

  ! CHECK: omp.loop private(@[[I_PRIV]] %{{.*}}#0 -> %[[ARG:.*]] : !fir.ref<i32>) {
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
  ! CHECK: omp.loop private(@{{.*}} %{{.*}}#0 -> %{{.*}}, @{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
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
  ! CHECK: omp.loop private(@[[DUMMY_PRIV]] %{{.*}}#0 -> %[[DUMMY_ARG:.*]], @{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
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
  ! CHECK: omp.loop bind(thread) private(@{{.*}} %{{.*}}#0 -> %{{.*}} : {{.*}}) {
  ! CHECK: }
  !$omp loop bind(thread)
  do i=1,10
   dummy = dummy + 1
  end do
  !$omp end loop
end subroutine
