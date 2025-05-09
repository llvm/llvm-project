! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[I_PRIVATE_TEST2:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[RES_PRIVATE_TEST2:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:       {type = private} @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:        {type = firstprivate} @[[RES_FIRSTPRIVATE:.*]] : i32 
! CHECK-SAME:   copy {
! CHECK:         hlfir.assign 

! CHECK-LABEL:  func.func @_QPomp_taskloop
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloopEi"}
! CHECK:          %[[I_VAL:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_RES:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskloopEres"}
! CHECK:          %[[RES_VAL:.*]]:2 = hlfir.declare %[[ALLOCA_RES]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:          %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:          %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:          omp.taskloop private(@[[RES_FIRSTPRIVATE]] %[[RES_VAL]]#0 -> %[[PRIV_RES:.*]], @[[I_PRIVATE]] %[[I_VAL]]#0 -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:            omp.loop_nest (%[[ARG2:.*]]) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:              %[[RES_DECL:.*]]:2 = hlfir.declare %[[PRIV_RES]] {uniq_name = "_QFomp_taskloopEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:              %[[I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I]] {uniq_name = "_QFomp_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:              hlfir.assign %[[ARG2]] to %[[I_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:              %[[LOAD_RES:.*]] = fir.load %[[RES_DECL]]#0 : !fir.ref<i32>
! CHECK:              %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:              %[[OUT_VAL:.*]] = arith.addi %[[LOAD_RES]], %[[C1_I32_1]] : i32
! CHECK:              hlfir.assign %[[OUT_VAL]] to %[[RES_DECL]]#0 : i32, !fir.ref<i32>
! CHECK:              omp.yield
! CHECK:            }
! CHECK:          }
! CHECK:          return
! CHECK:        }

subroutine omp_taskloop
  integer :: res, i
  !$omp taskloop
  do i = 1, 10
     res = res + 1
  end do
  !$omp end taskloop
end subroutine omp_taskloop


! CHECK-LABEL:  func.func @_QPomp_taskloop_private
! CHECK:           %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomp_taskloop_privateEi"}
! CHECK:           %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFomp_taskloop_privateEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           %[[ALLOCA_RES:.*]] = fir.alloca i32 {bindc_name = "res", uniq_name = "_QFomp_taskloop_privateEres"}
! CHECK:           %[[DECL_RES:.*]]:2 = hlfir.declare %[[ALLOCA_RES]] {uniq_name = "_QFomp_taskloop_privateEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
subroutine omp_taskloop_private
  integer :: res, i
! CHECK:           omp.taskloop private(@[[RES_PRIVATE_TEST2]] %[[DECL_RES]]#0 -> %[[ARG0:.*]], @[[I_PRIVATE_TEST2]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:             omp.loop_nest (%{{.*}}) : i32 = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
! CHECK:               %[[VAL1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFomp_taskloop_privateEres"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
  !$omp taskloop private(res)
  do i = 1, 10
! CHECK:               %[[LOAD_RES:.*]] = fir.load %[[VAL1]]#0 : !fir.ref<i32>
! CHECK:               %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:               %[[ADD_VAL:.*]] = arith.addi %[[LOAD_RES]], %[[C1_I32_1]] : i32
! CHECK:               hlfir.assign %[[ADD_VAL]] to %[[VAL1]]#0 : i32, !fir.ref<i32>
     res = res + 1
  end do
! CHECK:           return
! CHECK:         }
  !$omp end taskloop
end subroutine omp_taskloop_private
