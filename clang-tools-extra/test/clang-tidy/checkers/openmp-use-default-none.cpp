// RUN: %check_clang_tidy %s openmp-use-default-none %t -- -- -fopenmp=libomp -fopenmp-version=51
// RUN: %check_clang_tidy -std=c11 %s openmp-use-default-none %t -- -- -x c -fopenmp=libomp -fopenmp-version=51

//----------------------------------------------------------------------------//
// Null cases.
//----------------------------------------------------------------------------//

// 'for' directive can not have 'default' clause, no diagnostics.
void n0(const int a) {
#pragma omp for
  for (int b = 0; b < a; b++)
    ;
}

//----------------------------------------------------------------------------//
// Single-directive positive cases.
//----------------------------------------------------------------------------//

// 'parallel' directive.

// 'parallel' directive can have 'default' clause, but said clause is not
// specified, diagnosed.
void p0_0(void) {
#pragma omp parallel
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'parallel' does not specify 'default' clause, consider specifying 'default(none)' clause
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// with 'none' kind, all good.
void p0_1(void) {
#pragma omp parallel default(none)
  ;
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// but with 'shared' kind, which is not 'none', diagnose.
void p0_2(void) {
#pragma omp parallel default(shared)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'parallel' specifies 'default(shared)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:22: note: existing 'default' clause specified here
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// but with 'firstprivate' kind, which is not 'none', diagnose.
void p0_3(void) {
#pragma omp parallel default(firstprivate)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'parallel' specifies 'default(firstprivate)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:22: note: existing 'default' clause specified here
}

// 'task' directive.

// 'task' directive can have 'default' clause, but said clause is not
// specified, diagnosed.
void p1_0(void) {
#pragma omp task
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'task' does not specify 'default' clause, consider specifying 'default(none)' clause
}

// 'task' directive can have 'default' clause, and said clause specified,
// with 'none' kind, all good.
void p1_1(void) {
#pragma omp task default(none)
  ;
}

// 'task' directive can have 'default' clause, and said clause specified,
// but with 'shared' kind, which is not 'none', diagnose.
void p1_2(void) {
#pragma omp task default(shared)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'task' specifies 'default(shared)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:18: note: existing 'default' clause specified here
}

// 'task' directive can have 'default' clause, and said clause specified,
// but with 'firstprivate' kind, which is not 'none', diagnose.
void p1_3(void) {
#pragma omp task default(firstprivate)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'task' specifies 'default(firstprivate)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:18: note: existing 'default' clause specified here
}

// 'teams' directive. (has to be inside of 'target' directive)

// 'teams' directive can have 'default' clause, but said clause is not
// specified, diagnosed.
void p2_0(void) {
#pragma omp target
#pragma omp teams
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'teams' does not specify 'default' clause, consider specifying 'default(none)' clause
}

// 'teams' directive can have 'default' clause, and said clause specified,
// with 'none' kind, all good.
void p2_1(void) {
#pragma omp target
#pragma omp teams default(none)
  ;
}

// 'teams' directive can have 'default' clause, and said clause specified,
// but with 'shared' kind, which is not 'none', diagnose.
void p2_2(void) {
#pragma omp target
#pragma omp teams default(shared)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'teams' specifies 'default(shared)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:19: note: existing 'default' clause specified here
}

// 'teams' directive can have 'default' clause, and said clause specified,
// but with 'firstprivate' kind, which is not 'none', diagnose.
void p2_3(void) {
#pragma omp target
#pragma omp teams default(firstprivate)
  ;
  // CHECK-NOTES: :[[@LINE-2]]:1: warning: OpenMP directive 'teams' specifies 'default(firstprivate)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-3]]:19: note: existing 'default' clause specified here
}

// 'taskloop' directive.

// 'taskloop' directive can have 'default' clause, but said clause is not
// specified, diagnosed.
void p3_0(const int a) {
#pragma omp taskloop
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'taskloop' does not specify 'default' clause, consider specifying 'default(none)' clause
}

// 'taskloop' directive can have 'default' clause, and said clause specified,
// with 'none' kind, all good.
void p3_1(const int a) {
#pragma omp taskloop default(none) shared(a)
  for (int b = 0; b < a; b++)
    ;
}

// 'taskloop' directive can have 'default' clause, and said clause specified,
// but with 'shared' kind, which is not 'none', diagnose.
void p3_2(const int a) {
#pragma omp taskloop default(shared)
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'taskloop' specifies 'default(shared)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-4]]:22: note: existing 'default' clause specified here
}

// 'taskloop' directive can have 'default' clause, and said clause specified,
// but with 'firstprivate' kind, which is not 'none', diagnose.
void p3_3(const int a) {
#pragma omp taskloop default(firstprivate)
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'taskloop' specifies 'default(firstprivate)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-4]]:22: note: existing 'default' clause specified here
}

//----------------------------------------------------------------------------//
// Combined directives.
// Let's not test every single possible permutation/combination of directives,
// but just *one* combined directive. The rest will be the same.
//----------------------------------------------------------------------------//

// 'parallel' directive can have 'default' clause, but said clause is not
// specified, diagnosed.
void p4_0(const int a) {
#pragma omp parallel for
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'parallel for' does not specify 'default' clause, consider specifying 'default(none)' clause
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// with 'none' kind, all good.
void p4_1(const int a) {
#pragma omp parallel for default(none) shared(a)
  for (int b = 0; b < a; b++)
    ;
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// but with 'shared' kind, which is not 'none', diagnose.
void p4_2(const int a) {
#pragma omp parallel for default(shared)
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'parallel for' specifies 'default(shared)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-4]]:26: note: existing 'default' clause specified here
}

// 'parallel' directive can have 'default' clause, and said clause specified,
// but with 'firstprivate' kind, which is not 'none', diagnose.
void p4_3(const int a) {
#pragma omp parallel for default(firstprivate)
  for (int b = 0; b < a; b++)
    ;
  // CHECK-NOTES: :[[@LINE-3]]:1: warning: OpenMP directive 'parallel for' specifies 'default(firstprivate)' clause, consider using 'default(none)' clause instead
  // CHECK-NOTES: :[[@LINE-4]]:26: note: existing 'default' clause specified here
}
