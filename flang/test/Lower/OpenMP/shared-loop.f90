! RUN: bbc -emit-hlfir -fopenmp %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

! CHECK:  func.func @_QQmain() attributes
! CHECK:    %[[ALLOC_I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFEi"}
! CHECK:    %[[DECL_I:.*]]:2 = hlfir.declare %[[ALLOC_I]] {uniq_name = "_QFEi"} :
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
program omploop
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
end program
