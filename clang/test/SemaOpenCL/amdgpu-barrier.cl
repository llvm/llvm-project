// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -verify -cl-std=CL1.2 -triple amdgcn-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -verify -cl-std=CL2.0 -triple amdgcn-amd-amdhsa -Wno-unused-value %s

void foo() {
    typedef __amdgpu_named_workgroup_barrier_t SugaredArray[2];

    struct TestSimple {
        __amdgpu_named_workgroup_barrier_t x;
    };

    struct TestArray {
        __amdgpu_named_workgroup_barrier_t y[2];
    };

    struct TestSugared {
        SugaredArray z[2];
    };

    struct GoodWrapper {
        __amdgpu_named_workgroup_barrier_t x;
    };

    // Wrappers cannot have >1 field.
    struct WrapperHasTooManyFields { // expected-note {{'WrapperHasTooManyFields' is not a named barrier wrapper because it has more than one field}}
        __amdgpu_named_workgroup_barrier_t x; // expected-error {{fields of type '__amdgpu_named_workgroup_barrier_t' are only allowed in named barrier wrappers}}
        int other;
    };

    int n = 100;
    __amdgpu_named_workgroup_barrier_t v = 0; // expected-error {{initializing '__private __amdgpu_named_workgroup_barrier_t' with an expression of incompatible type 'int'}}
    int c = v; // expected-error {{initializing '__private int' with an expression of incompatible type '__private __amdgpu_named_workgroup_barrier_t'}}
    __amdgpu_named_workgroup_barrier_t k;
    int *ip = (int *)k; // expected-error {{operand of type '__amdgpu_named_workgroup_barrier_t' where arithmetic or pointer type is required}}
    void *vp = (void *)k; // expected-error {{operand of type '__amdgpu_named_workgroup_barrier_t' where arithmetic or pointer type is required}}
 }
