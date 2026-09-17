// RUN: %clang_cc1 -fopenmp -fopenmp-version=61 -ast-print %s | FileCheck %s
// expected-no-diagnostics

void test_ast_print() {
  int N = 10;
  int M = 20;
  constexpr int D = 3;

  // CHECK: #pragma omp target teams num_teams(dims(3):1,2,3) thread_limit(dims(2):10,20)
  #pragma omp target teams num_teams(dims(3): 1, 2, 3) thread_limit(dims(2): 10, 20)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(D):1,2,3) thread_limit(dims(D):10,20,30)
  #pragma omp target teams num_teams(dims(D): 1, 2, 3) thread_limit(dims(D): 10, 20, 30)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(2):N,N + 1) thread_limit(dims(1):M)
  #pragma omp target teams num_teams(dims(2): N, N + 1) thread_limit(dims(1): M)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(1):N) thread_limit(dims(1):M)
  #pragma omp target teams num_teams(dims(1): N) thread_limit(dims(1): M)
  {}

  // CHECK: #pragma omp parallel num_threads(dims(3): 1,2,3)
  #pragma omp parallel num_threads(dims(3): 1, 2, 3)
  {}

  // CHECK: #pragma omp parallel num_threads(strict: 10)
  #pragma omp parallel num_threads(strict: 10)
  {}

  // CHECK: #pragma omp parallel num_threads(strict,dims(2): N + 1,N)
  #pragma omp parallel num_threads(strict, dims(2): N + 1, N)
  {}

  // CHECK: #pragma omp parallel num_threads(strict,dims(2): 10,20)
  #pragma omp parallel num_threads(dims(2), strict: 10, 20)
  {}

  // CHECK: #pragma omp parallel num_threads(strict,dims(2): N,N + 1)
  #pragma omp parallel num_threads(strict, dims(2): N, N + 1)
  {}

  // CHECK: #pragma omp target parallel thread_limit(dims(2):10,20) num_threads(dims(2): N,N + 1)
  #pragma omp target parallel thread_limit(dims(2): 10, 20) num_threads(dims(2): N, N + 1)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(3):1,2,3) thread_limit(dims(2):10,20)
  // CHECK-NEXT: #pragma omp parallel num_threads(dims(2): N,N + 1)
  #pragma omp target teams num_teams(dims(3): 1, 2, 3) thread_limit(dims(2): 10, 20)
  #pragma omp parallel num_threads(dims(2): N, N + 1)
  {}

  // CHECK: #pragma omp target parallel thread_limit(dims(1):M) num_threads(strict,dims(2): 10,20)
  #pragma omp target parallel thread_limit(dims(1): M) num_threads(dims(2), strict: 10, 20)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(1):N) thread_limit(dims(1):M)
  // CHECK-NEXT: #pragma omp parallel num_threads(strict,dims(2): 10,20)
  #pragma omp target teams num_teams(dims(1): N) thread_limit(dims(1): M)
  #pragma omp parallel num_threads(dims(2), strict: 10, 20)
  {}
}

template <int D>
void template_test() {
  int arr[3] = {1, 2, 3};
  // CHECK: #pragma omp target teams num_teams(dims(3):arr[0],arr[1],arr[2]) thread_limit(dims(1):3)
  #pragma omp target teams num_teams(dims(3): arr[0], arr[1], arr[2]) thread_limit(dims(1): D)
  {}

  // CHECK: #pragma omp target teams num_teams(dims(3):arr[0],arr[1],arr[2]) thread_limit(dims(2):3,arr[0])
  #pragma omp target teams num_teams(dims(D): arr[0], arr[1], arr[2]) thread_limit(dims(2): D, arr[0])
  {}

  // CHECK: #pragma omp parallel num_threads(dims(3): arr[0],arr[1],arr[2])
  #pragma omp parallel num_threads(dims(3): arr[0], arr[1], arr[2])
  {}

  // CHECK: #pragma omp parallel num_threads(strict,dims(2): 3,arr[0])
  #pragma omp parallel num_threads(strict, dims(2): D, arr[0])
  {}

  // CHECK: #pragma omp parallel num_threads(strict,dims(2): 3,arr[0])
  #pragma omp parallel num_threads(dims(2), strict: D, arr[0])
  {}

  // CHECK: #pragma omp target parallel thread_limit(dims(1):3) num_threads(strict,dims(2): 3,arr[0])
  #pragma omp target parallel thread_limit(dims(1): D) num_threads(dims(2), strict: D, arr[0])
  {}

  // CHECK: #pragma omp target teams num_teams(dims(3):arr[0],arr[1],arr[2]) thread_limit(dims(1):3)
  // CHECK-NEXT: #pragma omp parallel num_threads(strict,dims(2): 3,arr[0])
  #pragma omp target teams num_teams(dims(3): arr[0], arr[1], arr[2]) thread_limit(dims(1): D)
  #pragma omp parallel num_threads(dims(2), strict: D, arr[0])
  {}
}

void call_templates() {
  template_test<3>();
}
