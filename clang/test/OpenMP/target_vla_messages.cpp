// PowerPC supports VLAs.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o %t-ppc-device.ll

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm-bc %s -o %t-ppc-host-ppc.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-unknown-unknown -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-ppc.bc -o %t-ppc-device.ll

// Nvidia GPUs don't support VLAs.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host-nvptx.bc
// RUN: %clang_cc1 -verify -DNO_VLA -fopenmp -x c++ -triple nvptx64-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host-nvptx.bc -o %t-nvptx-device.ll

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=nvptx64-nvidia-cuda -emit-llvm-bc %s -o %t-ppc-host-nvptx.bc

#ifndef NO_VLA
// expected-no-diagnostics
#endif

#pragma omp declare target
void declare(int arg) {
  int a[2];
#ifdef NO_VLA
  // expected-error@+2 {{variable length arrays are not supported for the current target}}
#endif
  int vla[arg];
}

void declare_parallel_reduction(int arg) {
  int a[2];

#pragma omp parallel reduction(+: a)
  { }

#pragma omp parallel reduction(+: a[0:2])
  { }

#ifdef NO_VLA
  // expected-error@+3 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+2 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp parallel reduction(+: a[0:arg])
  { }
}
#pragma omp end declare target

template <typename T>
void target_template(int arg) {
#pragma omp target
  {
#ifdef NO_VLA
    // expected-error@+2 {{variable length arrays are not supported for the current target}}
#endif
    T vla[arg];
  }
}

void target(int arg) {
#pragma omp target
  {
#ifdef NO_VLA
    // expected-error@+2 {{variable length arrays are not supported for the current target}}
#endif
    int vla[arg];
  }

#pragma omp target
  {
#pragma omp parallel
    {
#ifdef NO_VLA
    // expected-error@+2 {{variable length arrays are not supported for the current target}}
#endif
      int vla[arg];
    }
  }

#ifdef NO_VLA
    // expected-note@+2 {{in instantiation of function template specialization 'target_template<long>' requested here}}
#endif
  target_template<long>(arg);
}

void teams_reduction(int arg) {
  int a[2];
  int vla[arg];

#pragma omp target map(a)
#pragma omp teams reduction(+: a)
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla)
#pragma omp teams reduction(+: vla)
  { }

#pragma omp target map(a[0:2])
#pragma omp teams reduction(+: a[0:2])
  { }

#pragma omp target map(vla[0:2])
#pragma omp teams reduction(+: vla[0:2])
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(a[0:arg])
#pragma omp teams reduction(+: a[0:arg])
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla[0:arg])
#pragma omp teams reduction(+: vla[0:arg])
  { }
}

void parallel_reduction(int arg) {
  int a[2];
  int vla[arg];

#pragma omp target map(a)
#pragma omp parallel reduction(+: a)
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla)
#pragma omp parallel reduction(+: vla)
  { }

#pragma omp target map(a[0:2])
#pragma omp parallel reduction(+: a[0:2])
  { }

#pragma omp target map(vla[0:2])
#pragma omp parallel reduction(+: vla[0:2])
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(a[0:arg])
#pragma omp parallel reduction(+: a[0:arg])
  { }

#ifdef NO_VLA
  // expected-error@+4 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+3 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla[0:arg])
#pragma omp parallel reduction(+: vla[0:arg])
  { }
}

void for_reduction(int arg) {
  int a[2];
  int vla[arg];

#pragma omp target map(a)
#pragma omp parallel
#pragma omp for reduction(+: a)
  for (int i = 0; i < arg; i++) ;

#ifdef NO_VLA
  // expected-error@+5 {{cannot generate code for reduction on variable length array}}
  // expected-note@+4 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla)
#pragma omp parallel
#pragma omp for reduction(+: vla)
  for (int i = 0; i < arg; i++) ;

#pragma omp target map(a[0:2])
#pragma omp parallel
#pragma omp for reduction(+: a[0:2])
  for (int i = 0; i < arg; i++) ;

#pragma omp target map(vla[0:2])
#pragma omp parallel
#pragma omp for reduction(+: vla[0:2])
  for (int i = 0; i < arg; i++) ;

#ifdef NO_VLA
  // expected-error@+5 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+4 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(a[0:arg])
#pragma omp parallel
#pragma omp for reduction(+: a[0:arg])
  for (int i = 0; i < arg; i++) ;

#ifdef NO_VLA
  // expected-error@+5 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+4 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target map(vla[0:arg])
#pragma omp parallel
#pragma omp for reduction(+: vla[0:arg])
  for (int i = 0; i < arg; i++) ;
#ifdef NO_VLA
  // expected-error@+3 {{cannot generate code for reduction on array section, which requires a variable length array}}
  // expected-note@+2 {{variable length arrays are not supported for the current target}}
#endif
#pragma omp target reduction(+ : vla[0:arg])
  for (int i = 0; i < arg; i++) ;
}
