! REQUIRES: amdgpu-registered-target
! RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-device -munsafe-fp-atomics %s -o -|FileCheck -check-prefix=UNSAFE-FP-ATOMICS %s
! RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-device -fatomic-ignore-denormal-mode %s -o -|FileCheck -check-prefix=IGNORE-DENORMAL-MODE %s
! RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-device -fatomic-fine-grained-memory %s -o -|FileCheck -check-prefix=FINE-GRAINED-MEMORY %s
! RUN: %flang_fc1 -emit-llvm -triple amdgcn-amd-amdhsa -fopenmp -fopenmp-is-device -fatomic-remote-memory %s -o -|FileCheck -check-prefix=REMOTE-MEMORY %s
program test
    implicit none
    integer :: A, threads
    threads = 128
    A = 0
    !$omp target parallel num_threads(threads)
    !$omp atomic
    A =  A + 1
    !$omp end target parallel
end program test

!UNSAFE-FP-ATOMICS: %{{.*}} = atomicrmw add ptr {{.*}}, i32 1 monotonic, align 4, !amdgpu.ignore.denormal.mode !{{.*}}, !amdgpu.no.fine.grained.memory !{{.*}}, !amdgpu.no.remote.memory !{{.*}}
!IGNORE-DENORMAL-MODE: %{{.*}} = atomicrmw add ptr {{.*}}, i32 1 monotonic, align 4, !amdgpu.ignore.denormal.mode !{{.*}}, !amdgpu.no.fine.grained.memory !{{.*}}, !amdgpu.no.remote.memory !{{.*}}
!FINE-GRAINED-MEMORY: %{{.*}} = atomicrmw add ptr {{.*}}, i32 1 monotonic, align 4, !amdgpu.no.remote.memory !{{.*}}
!REMOTE-MEMORY: %{{.*}} = atomicrmw add ptr {{.*}}, i32 1 monotonic, align 4, !amdgpu.no.fine.grained.memory !{{.*}}
