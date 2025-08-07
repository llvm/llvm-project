// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -ferror-limit 100 -o - %s

// expected-warning@+1 {{directive spelling 'begin declare_target' is introduced in a later OpenMP version}}
#pragma omp begin declare_target
void f0();
// expected-warning@+1 {{directive spelling 'end declare_target' is introduced in a later OpenMP version}}
#pragma omp end declare_target

// expected-warning@+1 {{directive spelling 'begin declare_variant' is introduced in a later OpenMP version}}
#pragma omp begin declare_variant match(user={condition(true)})
void f1();
// expected-warning@+1 {{directive spelling 'end declare_variant' is introduced in a later OpenMP version}}
#pragma omp end declare_variant

int x;
// expected-warning@+1 {{directive spelling 'declare_target' is introduced in a later OpenMP version}}
#pragma omp declare_target(x)

struct A {
  int x, y;
};
// expected-warning@+1 {{directive spelling 'declare_mapper' is introduced in a later OpenMP version}}
#pragma omp declare_mapper(mymapper: A a) map(tofrom:a.x, a.y)
A add(A, A);
// expected-warning@+1 {{directive spelling 'declare_reduction' is introduced in a later OpenMP version}}
#pragma omp declare_reduction(+: A: omp_out = add(omp_in, omp_out))

// expected-warning@+1 {{directive spelling 'declare_simd' is introduced in a later OpenMP version}}
#pragma omp declare_simd
void f2();

void g3();
// expected-warning@+1 {{directive spelling 'declare_variant' is introduced in a later OpenMP version}}
#pragma omp declare_variant(g3) match(user={condition(true)})
void f3() {}

void fred() {
  #pragma omp parallel
  {
    // expected-warning@+1 {{directive spelling 'cancellation_point' is introduced in a later OpenMP version}}
    #pragma omp cancellation_point parallel
  }

  // expected-warning@+1 {{directive spelling 'target_data' is introduced in a later OpenMP version}}
  #pragma omp target_data map(tofrom: x)
  {}

  // expected-warning@+1 {{directive spelling 'target_enter_data' is introduced in a later OpenMP version}}
  #pragma omp target_enter_data map(to: x)
  // expected-warning@+1 {{directive spelling 'target_exit_data' is introduced in a later OpenMP version}}
  #pragma omp target_exit_data map(from: x)
  // expected-warning@+1 {{directive spelling 'target_update' is introduced in a later OpenMP version}}
  #pragma omp target_update from(x)
}

