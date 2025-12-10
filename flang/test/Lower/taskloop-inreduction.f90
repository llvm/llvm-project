! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private
! CHECK-SAME:        {type = private} @[[PRIVATE_I:.*]] : i32

! CHECK-LABEL:  omp.declare_reduction 
! CHECK-SAME:   @[[ADD_RED_I32:.*]] : i32 init {
! CHECK:       ^bb0(%{{.*}}: i32):
! CHECK:        %[[C0_I32:.*]] = arith.constant 0 : i32
! CHECK:        omp.yield(%[[C0_I32]] : i32)
! CHECK:     } combiner {
! CHECK:     ^bb0(%{{.*}}: i32, %{{.*}}: i32):
! CHECK:        %[[RES:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK:        omp.yield(%[[RES]] : i32)
! CHECK:     }

! CHECK-LABEL: func.func @_QPomp_taskloop_inreduction
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloop_inreductionEi"}
! CHECK:          %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloop_inreductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_X:.*]] = fir.alloca i32 {bindc_name = "x", uniq_name = "_QFomp_taskloop_inreductionEx"}
! CHECK:          %[[DECL_X:.*]]:2 = hlfir.declare %[[ALLOCA_X]] {uniq_name = "_QFomp_taskloop_inreductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[INIT_X:.*]] = arith.constant 0 : i32
! CHECK:          hlfir.assign %[[INIT_X]] to %[[DECL_X]]#0 : i32, !fir.ref<i32>
subroutine omp_taskloop_inreduction()
   integer x
   x = 0
   ! CHECK:        omp.taskloop in_reduction(@[[ADD_RED_I32]] 
   ! CHECK:        %[[DECL_X]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) private(@[[PRIVATE_I]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>) {
   ! CHECK:        %[[VAL_ARG1:.*]]:2 = hlfir.declare %[[ARG0]] 
   ! CHECK-SAME:   {uniq_name = "_QFomp_taskloop_inreductionEx"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
   !$omp taskloop in_reduction(+:x)
   do i = 1, 100
      ! CHECK: %[[X_VAL:.*]] = fir.load %[[VAL_ARG1]]#0 : !fir.ref<i32>
      ! CHECK: %[[ADD_VAL:.*]] = arith.addi %[[X_VAL]], %{{.*}} : i32
      x = x + 1
      ! CHECK: hlfir.assign %[[ADD_VAL]] to %[[VAL_ARG1]]#0 : i32, !fir.ref<i32>
   end do
   !$omp end taskloop
end subroutine omp_taskloop_inreduction
