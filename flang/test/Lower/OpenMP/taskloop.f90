! This test checks lowering of `Taskloop` directive and taskloop clauses "grainsize" and "numtasks".

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK:  func.func @_QPomp_taskloop() {
! CHECK:    %[[VAL_0:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloopEi"}
! CHECK:    %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_2:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskloopEres"}
! CHECK:    %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_2]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_4:.*]] = fir.alloca i32 {bindc_name = "res", pinned, uniq_name = "_QFomp_taskloopEres"}
! CHECK:    %[[VAL_5:.*]]:2 = hlfir.declare %[[VAL_4]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[VAL_6:.*]] = fir.load %[[VAL_3]]#0 : !fir.ref<i32>
! CHECK:    hlfir.assign %[[VAL_6]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:    %[[VAL_7:.*]] = fir.alloca i32 {bindc_name = "i", pinned, uniq_name = "_QFomp_taskloopEi"}
! CHECK:    %[[VAL_8:.*]]:2 = hlfir.declare %[[VAL_7]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:    %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:    %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:    %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:    omp.taskloop {
! CHECK:      omp.loop_nest (%arg0) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:        fir.store %arg0 to %[[VAL_8]]#1 : !fir.ref<i32>
! CHECK:        %[[VAL_9:.*]] = fir.load %[[VAL_5]]#0 : !fir.ref<i32>
! CHECK:        %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:        %[[VAL_10:.*]] = arith.addi %[[VAL_9]], %[[C1_I32_1]] : i32
! CHECK:        hlfir.assign %[[VAL_10]] to %[[VAL_5]]#0 : i32, !fir.ref<i32>
! CHECK:        omp.yield
! CHECK:      }
! CHECK:    }
! CHECK:    return
! CHECK:  }
subroutine omp_taskloop
  integer :: res, i
  !$omp taskloop
  do i = 1, 10
     res = res + 1
  end do
  !$omp end taskloop
end subroutine omp_taskloop

!CHECK-LABEL: func.func @_QPtest_grainsize()
subroutine test_grainsize
  integer :: res, i
  !CHECK: omp.taskloop grainsize(%{{.*}}: i32) {
  !$omp taskloop grainsize(10)
  do i = 1, 100
     !CHECK: arith.addi
     res = res + 1
  end do
  !$omp end taskloop
end subroutine test_grainsize

!CHECK-LABEL: func.func @_QPtest_grainsize_strict()
subroutine test_grainsize_strict
  integer :: res, i
  !CHECK: omp.taskloop grainsize(strict, %{{.*}}: i32) {
  !$omp taskloop grainsize(strict:10)
  do i = 1, 100
     !CHECK: arith.addi
     res = res + 1
  end do
  !$omp end taskloop
end subroutine

!CHECK-LABEL: func.func @_QPtest_num_tasks()
subroutine test_num_tasks
  integer :: res, i
  !CHECK: omp.taskloop num_tasks(%{{.*}}: i32) {
  !$omp taskloop num_tasks(10)
  do i = 1, 100
     !CHECK: arith.addi
     res = res + 1
  end do
  !$omp end taskloop
end subroutine

!CHECK-LABEL: func.func @_QPtest_num_tasks_strict()
subroutine test_num_tasks_strict
  integer :: res, i
  !CHECK: omp.taskloop num_tasks(strict, %{{.*}}: i32) {
  !$omp taskloop num_tasks(strict:10)
  do i = 1, 100
     !CHECK: arith.addi
     res = res + 1
  end do
  !$omp end taskloop
end subroutine