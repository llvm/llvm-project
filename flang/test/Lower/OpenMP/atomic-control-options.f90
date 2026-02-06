! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device -munsafe-fp-atomics %s -o - | FileCheck -check-prefix=UNSAFE-FP-ATOMICS %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device -fatomic-ignore-denormal-mode %s -o - | FileCheck -check-prefix=IGNORE-DENORMAL %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device -fatomic-fine-grained-memory %s -o - | FileCheck -check-prefix=FINE-GRAINED-MEMORY %s
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-device -fatomic-remote-memory %s -o - | FileCheck -check-prefix=REMOTE-MEMORY %s
program test
    implicit none
    integer :: A, B, threads
    threads = 128
    A = 0
    B = 0
    !UNSAFE-FP-ATOMICS: } {atomic_control = #omp.atomic_control<ignore_denormal_mode = true>}
    !IGNORE-DENORMAL: } {atomic_control = #omp.atomic_control<ignore_denormal_mode = true>}
    !FINE-GRAINED-MEMORY: } {atomic_control = #omp.atomic_control<fine_grained_memory = true>}
    !REMOTE-MEMORY: } {atomic_control = #omp.atomic_control<remote_memory = true>}
    !$omp target parallel num_threads(threads)
    !$omp atomic
    A =  A + 1
    !$omp end target parallel
    !UNSAFE-FP-ATOMICS: } {atomic_control = #omp.atomic_control<ignore_denormal_mode = true>}
    !IGNORE-DENORMAL: } {atomic_control = #omp.atomic_control<ignore_denormal_mode = true>}
    !FINE-GRAINED-MEMORY: } {atomic_control = #omp.atomic_control<fine_grained_memory = true>}
    !REMOTE-MEMORY: } {atomic_control = #omp.atomic_control<remote_memory = true>}
    !$omp target parallel num_threads(threads)
    !$omp atomic capture
        A = A + B
        B = A
    !$omp end atomic
    !$omp end target parallel
end program test
