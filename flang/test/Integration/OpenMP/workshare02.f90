!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

!RUN: %flang_fc1 -emit-mlir -fopenmp -O3 %s -o - | FileCheck %s --check-prefix MLIR

program test_ws_01
    implicit none
    real(8) :: arr_01(10), x
    arr_01 = [0.347,0.892,0.573,0.126,0.788,0.412,0.964,0.205,0.631,0.746]

    !$omp parallel workshare
        x = sum(arr_01)
    !$omp end parallel workshare
end program test_ws_01

! MLIR:  func.func @_QQmain
! MLIR:    omp.parallel {
!            [...]
! MLIR:      omp.wsloop {
! MLIR:        omp.loop_nest {{.*}}
!                [...]
! MLIR:          %[[SUM:.*]] = arith.addf {{.*}}
!                [...]
! MLIR:          omp.yield
! MLIR:        }
! MLIR:      }
! MLIR:      omp.barrier
! MLIR:      omp.terminator
! MLIR:      }
! MLIR:    return
! MLIR:    }
