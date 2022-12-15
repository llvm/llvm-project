// RUN: %clang_cc1 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -fopenmp-version=52 -DVERBOSE_MODE=1 -verify=omp52 -fnoopenmp-use-tls -ferror-limit 100 -fopenmp-targets=amdgcn-amd-amdhsa -o - %s

void fun();
void not_a_host_function();
#pragma omp declare target enter(fun) device_type(nohost) // omp52-note {{marked as 'device_type(nohost)' here}}
#pragma omp declare variant(not_a_host_function) match(device={kind(host)}) // omp52-error {{function with 'device_type(nohost)' is not available on host}}
void fun() {}
#pragma omp begin declare target device_type(nohost) // omp52-note {{marked as 'device_type(nohost)' here}}
void not_a_host_function() {}
#pragma omp end declare target
void failed_call_to_host_function() { fun(); } // omp52-error {{function with 'device_type(nohost)' is not available on host}}

void fun2();
void host_function();
#pragma omp declare target enter(fun2) device_type(nohost)
#pragma omp declare variant(host_function) match(device={kind(host)})
void fun2() {}
#pragma omp begin declare target device_type(host)
void host_function() {}
#pragma omp end declare target
void call_to_host_function() { fun2(); }
