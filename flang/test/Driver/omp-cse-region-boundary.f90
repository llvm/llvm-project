!This test checks that when compiling an OpenMP program for the target device
!CSE is not done across target op region boundaries. It also checks that when
!compiling for the host CSE is done.
!RUN: %flang_fc1 -fopenmp-is-target-device -emit-mlir -fopenmp %s -o - | fir-opt -cse | FileCheck %s -check-prefix=CHECK-DEVICE
!RUN: %flang_fc1 -emit-mlir -fopenmp %s -o - | fir-opt -cse | FileCheck %s -check-prefix=CHECK-HOST
!RUN: bbc -fopenmp-is-target-device -emit-fir -fopenmp %s -o - | fir-opt -cse | FileCheck %s -check-prefix=CHECK-DEVICE
!RUN: bbc -emit-fir -fopenmp %s -o - | fir-opt -cse | FileCheck %s -check-prefix=CHECK-HOST

!Constant should be present inside target region.
!CHECK-DEVICE: omp.target
!CHECK-DEVICE: arith.constant 10
!CHECK-DEVICE: omp.terminator

!Constant should not be present inside target region.
!CHECK-HOST: omp.target
!CHECK-HOST-NOT: arith.constant 10
!CHECK-HOST: omp.terminator

subroutine writeIndex(sum)
        integer :: sum
        integer :: myconst1
        integer :: myconst2
        myconst1 = 10
!$omp target map(from:myconst2)
        myconst2 = 10
!$omp end target
        sum = myconst2 + myconst2
end subroutine writeIndex
