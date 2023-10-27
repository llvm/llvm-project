// RUN: %clang_cc1 -fsyntax-only -fopenmp -fopenmp-version=50 -triple x86_64-unknown-unknown -verify=expected,omp50 %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp -triple x86_64-unknown-unknown -verify=expected,omp51 %s -Wuninitialized

// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -fopenmp-version=50 -triple x86_64-unknown-unknown -verify=expected,omp50 %s -Wuninitialized
// RUN: %clang_cc1 -fsyntax-only -fopenmp-simd -triple x86_64-unknown-unknown -verify=expected,omp51 %s -Wuninitialized

extern int omp_get_num_threads  (void);

int main(int argc, char **argv) {
  int A = 0;
#pragma omp parallel for order(concurrent)
  for (int i = 0; i < 10; ++i)
    omp_get_num_threads(); // omp51-error {{calls to OpenMP runtime API are not allowed within a region that corresponds to a construct with an order clause that specifies concurrent}}

#pragma omp parallel for order(reproducible:concurrent) // omp50-error {{expected 'concurrent' in OpenMP clause 'order'}}
  for (int i = 0; i < 10; ++i)
    omp_get_num_threads(); // omp51-error {{calls to OpenMP runtime API are not allowed within a region that corresponds to a construct with an order clause that specifies concurrent}}

#pragma omp parallel for order(unconstrained:concurrent) // omp50-error {{expected 'concurrent' in OpenMP clause 'order'}}
  for (int i = 0; i < 10; ++i)
    omp_get_num_threads(); // omp51-error {{calls to OpenMP runtime API are not allowed within a region that corresponds to a construct with an order clause that specifies concurrent}}

#pragma omp parallel for order(concurrent)
  for (int i = 0; i < 10; ++i) {
    for (int j = 0; j < 10; ++j) {
      omp_get_num_threads(); // omp51-error {{calls to OpenMP runtime API are not allowed within a region that corresponds to a construct with an order clause that specifies concurrent}}
    }
  }

#pragma omp parallel for order(concurrent)
  for (int i = 0; i < 10; ++i) {
#pragma omp atomic //omp51-error {{construct 'atomic' not allowed in a region associated with a directive with 'order' clause}}
      A++;
  }

#pragma omp parallel for order(reproducible: concurrent) // omp50-error {{expected 'concurrent' in OpenMP clause 'order'}}
  for (int i = 0; i < 10; ++i) {
#pragma omp target //omp51-error {{construct 'target' not allowed in a region associated with a directive with 'order' clause}}
      A++;
  }

#pragma omp parallel for order(unconstrained: concurrent) // omp50-error {{expected 'concurrent' in OpenMP clause 'order'}}
  for (int i = 0; i < 10; ++i) {
#pragma omp target //omp51-error {{construct 'target' not allowed in a region associated with a directive with 'order' clause}}
      A++;
  }
}
