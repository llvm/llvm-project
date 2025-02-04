! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! --- Check that with shared(i) the variable outside the parallel section
! --- is updated. 
! CHECK-LABEL:  func.func @_QPomploop()
! CHECK:    %[[ALLOC_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomploopEi"}
! CHECK:    %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOC_I]] {uniq_name = "_QFomploopEi"} :
! CHECK:    omp.parallel {
! CHECK:      omp.sections {
! CHECK:        omp.section {
! CHECK:          %[[RES:.*]]:2 = fir.do_loop %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG1:.*]] = 
! CHECK:            fir.store %[[ARG1]] to %[[DECL_I]]#1
! CHECK:            %[[UPDATE_ITER:.*]] = arith.addi %[[ARG0]], %{{.*}}
! CHECK:            %[[LOAD_I:.*]] = fir.load %[[DECL_I]]#1
! CHECK:            %[[RES_I:.*]] = arith.addi %[[LOAD_I]], %{{.*}}
! CHECK:            fir.result %[[UPDATE_ITER]], %[[RES_I]]
! CHECK:          }
! CHECK:          fir.store %[[RES]]#1 to %[[DECL_I]]#1
! CHECK:          omp.terminator
! CHECK:        }
! CHECK:        omp.terminator
! CHECK:      }
! CHECK:      return
! CHECK:    }
subroutine omploop
  implicit none
  integer :: i, j
  i = 1
  j = 0
  !$omp parallel shared(i)
    !$omp sections
      do i=1,10
         j = j + i
      end do
    !$omp end sections
  !$omp end parallel
end subroutine

! --- Check that with default(shared) the variable outside the parallel section
! --- is NOT updated (i is private to the omp.parallel code)
! CHECK-LABEL:  func.func @_QPomploop2()
! CHECK:    %[[ALLOC_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomploop2Ei"}
! CHECK:    %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOC_I]] {uniq_name = "_QFomploop2Ei"} :
! CHECK:    omp.parallel {
! CHECK:      %[[ALLOC_PRIV_I:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
! CHECK:      %[[DECL_PRIV_I:.*]]:2 = hlfir.declare %[[ALLOC_PRIV_I]]
! CHECK:      omp.sections {
! CHECK:        omp.section {
! CHECK:          %[[RES:.*]]:2 = fir.do_loop %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG1:.*]] = 
! CHECK-NOT:            fir.store %[[ARG1]] to %[[DECL_I]]#1
! CHECK:            fir.store %[[ARG1]] to %[[DECL_PRIV_I]]#1
! CHECK:            %[[UPDATE_ITER:.*]] = arith.addi %[[ARG0]], %{{.*}}
! CHECK:            %[[LOAD_I:.*]] = fir.load %[[DECL_PRIV_I]]#1
! CHECK:            %[[RES_I:.*]] = arith.addi %[[LOAD_I]], %{{.*}}
! CHECK:            fir.result %[[UPDATE_ITER]], %[[RES_I]]
! CHECK:          }
! CHECK:          fir.store %[[RES]]#1 to %[[DECL_PRIV_I]]#1
! CHECK:          omp.terminator
! CHECK:        }
! CHECK:        omp.terminator
! CHECK:      }
! CHECK:      return
! CHECK:    }
subroutine omploop2
  implicit none
  integer :: i, j
  i = 1
  j = 0
  !$omp parallel default(shared)
    !$omp sections
      do i=1,10
         j = j + i
      end do
    !$omp end sections
  !$omp end parallel
end subroutine


! --- Check that with no data-sharing the variable outside the parallel section
! --- is NOT updated (i is private to the omp.parallel code)
! CHECK-LABEL:  func.func @_QPomploop3()
! CHECK:    %[[ALLOC_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFomploop3Ei"}
! CHECK:    %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOC_I]] {uniq_name = "_QFomploop3Ei"} :
! CHECK:    omp.parallel {
! CHECK:      %[[ALLOC_PRIV_I:.*]] = fir.alloca i32 {bindc_name = "i", pinned}
! CHECK:      %[[DECL_PRIV_I:.*]]:2 = hlfir.declare %[[ALLOC_PRIV_I]]
! CHECK:      omp.sections {
! CHECK:        omp.section {
! CHECK:          %[[RES:.*]]:2 = fir.do_loop %[[ARG0:.*]] = %{{.*}} to %{{.*}} step %{{.*}} iter_args(%[[ARG1:.*]] = 
! CHECK-NOT:            fir.store %[[ARG1]] to %[[DECL_I]]#1
! CHECK:            fir.store %[[ARG1]] to %[[DECL_PRIV_I]]#1
! CHECK:            %[[UPDATE_ITER:.*]] = arith.addi %[[ARG0]], %{{.*}}
! CHECK:            %[[LOAD_I:.*]] = fir.load %[[DECL_PRIV_I]]#1
! CHECK:            %[[RES_I:.*]] = arith.addi %[[LOAD_I]], %{{.*}}
! CHECK:            fir.result %[[UPDATE_ITER]], %[[RES_I]]
! CHECK:          }
! CHECK:          fir.store %[[RES]]#1 to %[[DECL_PRIV_I]]#1
! CHECK:          omp.terminator
! CHECK:        }
! CHECK:        omp.terminator
! CHECK:      }
! CHECK:      return
! CHECK:    }
subroutine omploop3
  implicit none
  integer :: i, j
  i = 1
  j = 0
  !$omp parallel
    !$omp sections
      do i=1,10
         j = j + i
      end do
    !$omp end sections
  !$omp end parallel
end subroutine


 
