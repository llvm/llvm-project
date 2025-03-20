! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda %s -o - | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa,nvptx64-nvidia-cuda %s -o - | FileCheck %s

! This test checks the addition of the omp.target_triples attribute when the
! -fopenmp-targets option is set

!CHECK:      module attributes {
!CHECK-SAME: omp.target_triples = ["amdgcn-amd-amdhsa", "nvptx64-nvidia-cuda"]
program targets
end program targets
