! This test checks lowering of OpenMP parallel masked taskloop Directive.

! RUN: bbc -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=50 -o - %s 2>&1 | FileCheck %s

! CHECK-LABEL:  omp.private {type = private} 
! CHECK-SAME:        @[[I_PRIVATE:.*]] : i32
! CHECK-LABEL:    func.func @_QPtest_parallel_master_taskloop() {
! CHECK:          %[[VAL0:.*]] = fir.dummy_scope : !fir.dscope
! CHECK:          %[[ALLOCA_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_parallel_master_taskloopEi"}
! CHECK:          %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOCA_I]] {uniq_name = "_QFtest_parallel_master_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          %[[ADDR_J:.*]] = fir.address_of(@_QFtest_parallel_master_taskloopEj) : !fir.ref<i32>
! CHECK:          %[[DECL_J:.*]]:2 = hlfir.declare %[[ADDR_J]] {uniq_name = "_QFtest_parallel_master_taskloopEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:          omp.parallel {
! CHECK:            omp.masked {
! CHECK:              %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:              %[[C10_I32:.*]] = arith.constant 10 : i32
! CHECK:              %[[C1_I32_0:.*]] = arith.constant 1 : i32
! CHECK:              omp.taskloop private(@[[I_PRIVATE]] %[[DECL_I]]#0 -> %[[ARG0:.*]] : !fir.ref<i32>) {
! CHECK:                omp.loop_nest (%[[ARG1:.*]]) : i32 = (%[[C1_I32]]) to (%[[C10_I32]]) inclusive step (%c1_i32_0) {
! CHECK:                  %[[VAL1:.*]]:2 = hlfir.declare %[[ARG0]] {uniq_name = "_QFtest_parallel_master_taskloopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:                  hlfir.assign %[[ARG1]] to %[[VAL1]]#0 : i32, !fir.ref<i32>
! CHECK:                  %[[LOAD_J:.*]] = fir.load %[[DECL_J]]#0 : !fir.ref<i32>
! CHECK:                  %c1_i32_1 = arith.constant 1 : i32
! CHECK:                  %[[RES_ADD:.*]] = arith.addi %[[LOAD_J]], %c1_i32_1 : i32
! CHECK:                  hlfir.assign %[[RES_ADD]] to %[[DECL_J]]#0 : i32, !fir.ref<i32>
! CHECK:                  omp.yield
! CHECK:                }
! CHECK:              }
! CHECK:              omp.terminator
! CHECK:            }
! CHECK:            omp.terminator
! CHECK:          }
! CHECK:          return
! CHECK:        }
! CHECK:        fir.global internal @_QFtest_parallel_master_taskloopEj : i32 {
! CHECK:          %[[C1_I32:.*]] = arith.constant 1 : i32
! CHECK:          fir.has_value %[[C1_I32]] : i32
! CHECK:        }

subroutine test_parallel_master_taskloop
  integer :: i, j = 1
  !$omp parallel masked taskloop
  do i=1,10
   j = j + 1
  end do
  !$omp end parallel masked taskloop 
end subroutine
