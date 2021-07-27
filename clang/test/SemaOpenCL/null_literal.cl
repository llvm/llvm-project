// RUN: %clang_cc1 -cl-std=CL1.0 -fdeclare-opencl-builtins -finclude-default-header -verify %s
// RUN: %clang_cc1 -cl-std=CL1.1 -fdeclare-opencl-builtins -finclude-default-header -verify %s
// RUN: %clang_cc1 -cl-std=CL1.2 -fdeclare-opencl-builtins -finclude-default-header -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -verify %s
// RUN: %clang_cc1 -cl-std=clc++ -fdeclare-opencl-builtins -finclude-default-header -verify %s

void foo(){

global int* ptr1 = NULL;

#if defined(__OPENCL_CPP_VERSION__)
// expected-error@+2{{cannot initialize a variable of type '__global int *__private' with an rvalue of type '__global void *'}}
#endif
global int* ptr2 = (global void*)0;

constant int* ptr3 = NULL;

#if defined(__OPENCL_CPP_VERSION__)
// expected-error@+4{{cannot initialize a variable of type '__constant int *__private' with an rvalue of type '__global void *'}}
#else
// expected-error@+2{{initializing '__constant int *__private' with an expression of type '__global void *' changes address space of pointer}}
#endif
constant int* ptr4 = (global void*)0;

#if __OPENCL_C_VERSION__ == CL_VERSION_2_0
// Accept explicitly pointer to generic address space in OpenCL v2.0.
global int* ptr5 = (generic void*)0;
#endif

#if defined(__OPENCL_CPP_VERSION__)
// expected-error@+4{{cannot initialize a variable of type '__global int *__private' with an rvalue of type '__local void *'}}
#else
// expected-error@+2{{initializing '__global int *__private' with an expression of type '__local void *' changes address space of pointer}}
#endif
global int* ptr6 = (local void*)0;

bool cmp = ptr1 == NULL;

#if defined(__OPENCL_CPP_VERSION__)
// expected-error@+4{{comparison of distinct pointer types ('__global int *' and '__local void *')}}
#else
// expected-error@+2{{comparison between  ('__global int *' and '__local void *') which are pointers to non-overlapping address spaces}}
#endif
cmp = ptr1 == (local void*)0;

cmp = ptr3 == NULL;

}
