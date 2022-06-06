! REQUIRES: plugins, examples, shell

! RUN: %flang_fc1 -load %llvmshlibdir/flangOmpReport.so -plugin flang-omp-report -fopenmp  %s -o - | FileCheck %s

! Check for IN_REDUCTION() clause on OpenMP constructs

subroutine omp_in_reduction_taskgroup()
    integer :: z, i
    !$omp taskgroup task_reduction(+:z)
    !$omp task in_reduction(+:z)
        z = z + 5
    !$omp end task

    !$omp taskloop in_reduction(+:z)
        do i=1,10
            z = z * 5
        end do
    !$omp end taskloop 
    !$omp end taskgroup
end subroutine omp_in_reduction_taskgroup

!CHECK: - file:         {{.*}}
!CHECK:   line:         10
!CHECK:   construct:    task
!CHECK:   clauses:
!CHECK:     - clause:   in_reduction
!CHECK:       details:  '+:z'
!CHECK: - file:         {{.*}}
!CHECK:   line:         14
!CHECK:   construct:    taskloop
!CHECK:   clauses:
!CHECK:     - clause:   in_reduction
!CHECK:       details:  '+:z'
!CHECK: - file:         {{.*}}
!CHECK:   line:         9
!CHECK:   construct:    taskgroup
!CHECK:   clauses:
!CHECK:      - clause:  task_reduction
!CHECK:        details: '+:z'

subroutine omp_in_reduction_parallel()
    integer :: z
    !$omp parallel reduction(+:z)
        !$omp taskloop simd in_reduction(+:z)
            do i=1,10
                z = z * 5
            end do
        !$omp end taskloop simd
    !$omp end parallel
end subroutine omp_in_reduction_parallel

!CHECK: - file:         {{.*}}
!CHECK:   line:         44
!CHECK:   construct:    taskloop simd
!CHECK:   clauses:
!CHECK:     - clause:   in_reduction
!CHECK:       details:  '+:z'
!CHECK:  - file:        {{.*}}
!CHECK:    line:        43
!CHECK:    construct:   parallel
!CHECK:    clauses:
!CHECK:      - clause:  reduction
!CHECK:        details: '+:z'
