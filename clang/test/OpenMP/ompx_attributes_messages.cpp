// RUN: %clang_cc1 -verify=expected -fopenmp -ferror-limit 100 -o - -std=c++11 %s -Wuninitialized

void bad() {
  #pragma omp target data ompx_attribute() //  expected-error {{unexpected OpenMP clause 'ompx_attribute' in directive '#pragma omp target data'}}
  #pragma omp target data ompx_attribute(__attribute__((launch_bounds(1, 2)))) //  expected-error {{unexpected OpenMP clause 'ompx_attribute' in directive '#pragma omp target data'}} expected-error {{expected at least one 'map', 'use_device_ptr', or 'use_device_addr' clause for '#pragma omp target data'}}

  #pragma omp target ompx_attribute()
  {}
  #pragma omp target ompx_attribute(__attribute__(()))
  {}
  #pragma omp target ompx_attribute(__attribute__((pure))) //  expected-warning {{'ompx_attribute' clause only allows 'amdgpu_flat_work_group_size', 'amdgpu_waves_per_eu', and 'launch_bounds'; 'pure' is ignored}}
  {}
  #pragma omp target ompx_attribute(__attribute__((pure,amdgpu_waves_per_eu(1, 2), const))) //  expected-warning {{'ompx_attribute' clause only allows 'amdgpu_flat_work_group_size', 'amdgpu_waves_per_eu', and 'launch_bounds'; 'pure' is ignored}} expected-warning {{'ompx_attribute' clause only allows 'amdgpu_flat_work_group_size', 'amdgpu_waves_per_eu', and 'launch_bounds'; 'const' is ignored}}
  {}
  #pragma omp target ompx_attribute(__attribute__((amdgpu_waves_per_eu()))) //  expected-error {{'amdgpu_waves_per_eu' attribute takes at least 1 argument}}
  {}
  #pragma omp target ompx_attribute(__attribute__((amdgpu_waves_per_eu(1, 2, 3)))) //  expected-error {{'amdgpu_waves_per_eu' attribute takes no more than 2 arguments}}
  {}
  #pragma omp target ompx_attribute(__attribute__((amdgpu_flat_work_group_size(1)))) //  expected-error {{'amdgpu_flat_work_group_size' attribute requires exactly 2 arguments}}
  {}
  #pragma omp target ompx_attribute(__attribute__((amdgpu_flat_work_group_size(1, 2, 3,)))) //  expected-error {{expected expression}}
  {}
  #pragma omp target ompx_attribute([[clang::amdgpu_waves_per_eu(1, 2, 3)]]) //  expected-error {{'amdgpu_waves_per_eu' attribute takes no more than 2 arguments}}
  {}
  #pragma omp target ompx_attribute([[clang::unknown]]) //  expected-warning {{'ompx_attribute' clause only allows 'amdgpu_flat_work_group_size', 'amdgpu_waves_per_eu', and 'launch_bounds'; 'unknown' is ignored}}
  {}
  #pragma omp target ompx_attribute(baz) //  expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1))))
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(bad)))) //  expected-error {{'launch_bounds' attribute requires parameter 0 to be an integer constant}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, //  expected-error {{expected expression}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, 2 //  expected-error {{expected ')'}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, 2) //  expected-error {{expected ')'}} expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, 2)) //  expected-error {{expected ')'}} expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, 2))) //  expected-error {{expected ')'}} expected-note {{to match this '('}}
  {}
  #pragma omp target ompx_attribute(__attribute__((launch_bounds(1, -3)))) //  expected-warning {{'launch_bounds' attribute parameter 1 is negative and will be ignored}}
  {}
  #pragma omp target ompx_attribute(__attribute__((amdgpu_waves_per_eu(10, 1)))) //  expected-error {{'amdgpu_waves_per_eu' attribute argument is invalid: min must not be greater than max}}
  {}
}
