// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -fopenmp -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -target-feature +sme2 -fopenmp -fsyntax-only -verify=expected-cpp -x c++ %s

int compute(int);

void streaming_openmp_captured_region(int * out) __arm_streaming {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in streaming functions}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in streaming functions}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

__arm_locally_streaming void locally_streaming_openmp_captured_region(int * out) {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in streaming functions}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in streaming functions}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

void za_state_captured_region(int * out) __arm_inout("za") {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in functions with ZA state}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in functions with ZA state}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

__arm_new("za") void new_za_state_captured_region(int * out) {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in functions with ZA state}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in functions with ZA state}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

void zt0_state_openmp_captured_region(int * out) __arm_inout("zt0") {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in functions with ZT0 state}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in functions with ZT0 state}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

__arm_new("zt0") void new_zt0_state_openmp_captured_region(int * out) {
  // expected-error@+2 {{OpenMP captured regions are not yet supported in functions with ZT0 state}}
  // expected-cpp-error@+1 {{OpenMP captured regions are not yet supported in functions with ZT0 state}}
  #pragma omp parallel for num_threads(32)
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

/// OpenMP directives that don't create a captured region are okay:

void streaming_function_openmp(int * out) __arm_streaming __arm_inout("za", "zt0") {
  #pragma omp unroll full
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

__arm_locally_streaming void locally_streaming_openmp(int * out) __arm_inout("za", "zt0") {
  #pragma omp unroll full
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}

__arm_new("za", "zt0") void arm_new_openmp(int * out) {
  #pragma omp unroll full
  for (int ci = 0; ci < 8; ci++) {
    out[ci] = compute(ci);
  }
}
