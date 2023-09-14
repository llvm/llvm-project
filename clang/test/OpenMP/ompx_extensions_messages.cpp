// RUN: %clang_cc1 -verify=expected -fopenmp -fno-openmp-extensions -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

void bad() {
  #pragma omp taskgraph //  expected-error {{Using extension directive 'taskgraph' in #pragma omp instead of #pragma ompx}}
  {} 
  #pragma ompx taskgraph //  expected-warning {{OpenMP Extensions not enabled. Ignoring OpenMP Extension Directive '#pragma ompx taskgraph'}}
  {}
  #pragma omp target ompx_attribute()  //  expected-warning {{OpenMP Extensions not enabled. Ignoring OpenMP Extension Clause 'ompx_attribute'}}
  {} 
  #pragma omp target ompx_dyn_cgroup_mem(1024) //  expected-warning {{OpenMP Extensions not enabled. Ignoring OpenMP Extension Clause 'ompx_dyn_cgroup_mem'}}
  {} 
}
