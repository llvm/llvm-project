// RUN: %clang_cc1 -triple x86_64 -verify=expected,dev \
// RUN:            -verify-ignore-unexpected=note \
// RUN:            -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -o - %s
// RUN: %clang_cc1 -triple x86_64 -verify -verify-ignore-unexpected=note\
// RUN:            -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -o - -x c++ %s
// RUN: %clang_cc1 -triple x86_64 -verify=dev -verify-ignore-unexpected=note\
// RUN:            -fcuda-is-device -o - %s

#if __CUDA__
#include "Inputs/cuda.h"
__device__ void cu_devf();
#endif

void bazz() {}
#pragma omp declare target to(bazz) device_type(nohost)
void bazzz() {bazz();}
#pragma omp declare target to(bazzz) device_type(nohost)
void any() {bazz();} // expected-error {{function with 'device_type(nohost)' is not available on host}}
void host1() {bazz();} // expected-error {{function with 'device_type(nohost)' is not available on host}}
#pragma omp declare target to(host1) device_type(host)
void host2() {bazz();} // expected-error {{function with 'device_type(nohost)' is not available on host}}
#pragma omp declare target to(host2)
void device() {host1();}
#pragma omp declare target to(device) device_type(nohost)
void host3() {host1();}
#pragma omp declare target to(host3)

#pragma omp declare target
void any1() {any();}
void any2() {host1();}
void any3() {device();} // expected-error {{function with 'device_type(nohost)' is not available on host}}
void any4() {any2();}
#pragma omp end declare target

void any5() {any();}
void any6() {host1();}
void any7() {device();} // expected-error {{function with 'device_type(nohost)' is not available on host}}
void any8() {any2();}

#if __CUDA__
void cu_hostf() { cu_devf(); } // dev-error {{no matching function for call to 'cu_devf'}}
__device__ void cu_devf2() { cu_hostf(); } // dev-error{{no matching function for call to 'cu_hostf'}}
#endif
