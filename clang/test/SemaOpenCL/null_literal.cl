// RUN: %clang_cc1 -verify %s
// RUN: %clang_cc1 -cl-std=CL2.0 -DCL20 -verify %s

#define NULL ((void*)0)

void foo(){

global int* ptr1 = NULL;

global int* ptr2 = (global void*)0;

constant int* ptr3 = NULL;

constant int* ptr4 = (global void*)0; // expected-error{{initializing '__constant int *__private' with an expression of type '__global void *' changes address space of pointer}}

#ifdef CL20
// Accept explicitly pointer to generic address space in OpenCL v2.0.
global int* ptr5 = (generic void*)0;
#endif

global int* ptr6 = (local void*)0; // expected-error{{initializing '__global int *__private' with an expression of type '__local void *' changes address space of pointer}}

bool cmp = ptr1 == NULL;

cmp = ptr1 == (local void*)0; // expected-error{{comparison between  ('__global int *' and '__local void *') which are pointers to non-overlapping address spaces}}

cmp = ptr3 == NULL;

}
