// RUN: not %clang_cc1 -triple x86_64-linux-gnu -fopenmp -mllvm --ns-disallow-unsupported-omp-features %s 2>&1 | FileCheck %s

void test() {
// CHECK: OpenMP clause 'proc_bind' is not supported on NextSilicon system
    #pragma omp parallel proc_bind(close)
    {
    }

// CHECK: OpenMP directive 'teams' is not supported on NextSilicon system
    #pragma omp teams
    {
    }

// CHECK: OpenMP directive 'sections' is not supported on NextSilicon system
// CHECK: OpenMP directive 'section' is not supported on NextSilicon system
    #pragma omp sections
    {
        #pragma omp section
    }

// CHECK: OpenMP directive 'cancellation point' is not supported on NextSilicon system
    #pragma omp cancellation point
    {
    }

// CHECK: OpenMP directive 'cancel' is not supported on NextSilicon system
    #pragma omp parallel
    {
        #pragma omp cancel
    }

// CHECK: OpenMP directive 'target teams' is not supported on NextSilicon system
    #pragma omp target teams
    {
    }

// CHECK: OpenMP directive 'task' is not supported on NextSilicon system
    #pragma omp task
    {
    }

// CHECK: OpenMP directive 'taskwait' is not supported on NextSilicon system
    #pragma omp taskwait
    {
    }

// CHECK: OpenMP directive 'taskgroup' is not supported on NextSilicon system
    #pragma omp taskgroup
    {
    }

// CHECK: OpenMP directive 'taskyield' is not supported on NextSilicon system
    #pragma omp taskyield
    {
    }
}

