! This test checks lowering of OpenMP masked taskloop Directive.

! RUN:  bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN:  %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private {type = private} 
! CHECK-SAME:        @[[I_PRIVATE:.*]] : i32

! CHECK-LABEL:  omp.private 
! CHECK-SAME:      {type = firstprivate} @[[J_FIRSTPRIVATE:.*]] : i32 
! CHECK-SAME:      copy {
! CHECK:            hlfir.assign 

! CHECK-LABEL:  func.func @_QPtest_masked_taskloop() {
! CHECK:          %[[VAL_0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_masked_taskloopEi"}
! CHECK:          %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] 
! CHECK-SAME:         {uniq_name = "_QFtest_masked_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ALLOCA_J:.*]] = fir.address_of(@_QFtest_masked_taskloopEj) : !fir.ref<i32>
! CHECK:          %[[DECL_J:.*]]:2 = hlfir.declare %[[ALLOCA_J]] {uniq_name = "_QFtest_masked_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          omp.masked {
! CHECK:            %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:            %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:            %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:            omp.taskloop private(
! CHECK-SAME:          @[[J_FIRSTPRIVATE]] %[[DECL_J]]#0 -> %[[ARG0:.*]], @[[I_PRIVATE]] %[[DECL_I]]#0 -> %[[ARG1:.*]] : !fir.ref<i32>, !fir.ref<i32>) {
! CHECK:              omp.loop_nest (%arg2) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%[[C1_I32_0]]) {
! CHECK:                %[[VAL1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFtest_masked_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                %[[VAL2:.*]]:2 = hlfir.declare %[[ARG1]] {uniq_name = "_QFtest_masked_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                hlfir.assign %arg2 to %[[VAL2]]#0 : i32, !fir.ref<i32>
! CHECK:                %[[LOAD_J:.*]] = fir.load %[[VAL1]]#0 : !fir.ref<i32>
! CHECK:                %[[C1_I32_1:.*]] = arith.constant 1 : i32
! CHECK:                %[[RES_J:.*]] = arith.addi %[[LOAD_J]], %[[C1_I32_1]] : i32
! CHECK:                hlfir.assign %[[RES_J]] to %[[VAL1]]#0 : i32, !fir.ref<i32>
! CHECK:                omp.yield
! CHECK:              }
! CHECK:            }
! CHECK:            omp.terminator
! CHECK:          }
! CHECK:          return
! CHECK:        }
! CHECK:        fir.global internal @_QFtest_masked_taskloopEj : i32 {
! CHECK:          %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:          fir.has_value %[[C1_I32]] : i32
! CHECK:        }


subroutine test_masked_taskloop
  integer :: i, j = 1
  !$omp masked taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end masked taskloop 
end subroutine
