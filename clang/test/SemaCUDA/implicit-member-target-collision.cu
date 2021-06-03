// RUN: %clang_cc1 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

//------------------------------------------------------------------------------
// Test 1: collision between two bases

struct A1_with_host_ctor {
  A1_with_host_ctor() {}
};

struct B1_with_device_ctor {
  __device__ B1_with_device_ctor() {}
};

struct C1_with_collision : A1_with_host_ctor, B1_with_device_ctor {
};

// expected-note@-3 {{candidate constructor (the implicit default constructor}} not viable
// expected-note@-4 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}
// expected-note@-5 {{candidate constructor (the implicit copy constructor}} not viable
// expected-note@-6 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo1() {
  C1_with_collision c; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 2: collision between two fields

struct C2_with_collision {
  A1_with_host_ctor aa;
  B1_with_device_ctor bb;
};

// expected-note@-5 {{candidate constructor (the implicit default constructor}} not viable
// expected-note@-6 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}
// expected-note@-7 {{candidate constructor (the implicit copy constructor}} not viable
// expected-note@-8 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo2() {
  C2_with_collision c; // expected-error {{no matching constructor}}

}

//------------------------------------------------------------------------------
// Test 3: collision between a field and a base

struct C3_with_collision : A1_with_host_ctor {
  B1_with_device_ctor bb;
};

// expected-note@-4 {{candidate constructor (the implicit default constructor}} not viable
// expected-note@-5 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}
// expected-note@-6 {{candidate constructor (the implicit copy constructor}} not viable
// expected-note@-7 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo3() {
  C3_with_collision c; // expected-error {{no matching constructor}}
}
