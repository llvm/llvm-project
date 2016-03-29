// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s

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

// expected-note@-3 {{candidate constructor (the implicit default constructor) not viable}}
// expected-note@-4 {{implicit default constructor inferred target collision: call to both __host__ and __device__ members}}
// expected-note@-5 {{candidate constructor (the implicit copy constructor) not viable}}
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
// expected-note@-8 {{candidate constructor (the implicit move constructor}} not viable

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
// expected-note@-7 {{candidate constructor (the implicit move constructor}} not viable

void hostfoo3() {
  C3_with_collision c; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 4: collision on resolving a copy ctor

struct A4_with_host_copy_ctor {
  A4_with_host_copy_ctor() {}
  A4_with_host_copy_ctor(const A4_with_host_copy_ctor&) {}
};

struct B4_with_device_copy_ctor {
  B4_with_device_copy_ctor() {}
  __device__ B4_with_device_copy_ctor(const B4_with_device_copy_ctor&) {}
};

struct C4_with_collision : A4_with_host_copy_ctor, B4_with_device_copy_ctor {
};

// expected-note@-3 {{copy constructor of 'C4_with_collision' is implicitly deleted because base class 'B4_with_device_copy_ctor' has no copy constructor}}

void hostfoo4() {
  C4_with_collision c;
  C4_with_collision c2 = c; // expected-error {{call to implicitly-deleted copy constructor of 'C4_with_collision'}}
}

//------------------------------------------------------------------------------
// Test 5: collision on resolving a move ctor

struct A5_with_host_move_ctor {
  A5_with_host_move_ctor() {}
  A5_with_host_move_ctor(A5_with_host_move_ctor&&) {}
// expected-note@-1 {{copy constructor is implicitly deleted because 'A5_with_host_move_ctor' has a user-declared move constructor}}
};

struct B5_with_device_move_ctor {
  B5_with_device_move_ctor() {}
  __device__ B5_with_device_move_ctor(B5_with_device_move_ctor&&) {}
};

struct C5_with_collision : A5_with_host_move_ctor, B5_with_device_move_ctor {
};
// expected-note@-2 {{deleted}}

void hostfoo5() {
  C5_with_collision c;
  // What happens here:
  // This tries to find the move ctor. Since the move ctor is deleted due to
  // collision, it then looks for a copy ctor. But copy ctors are implicitly
  // deleted when move ctors are declared explicitly.
  C5_with_collision c2(static_cast<C5_with_collision&&>(c)); // expected-error {{call to implicitly-deleted}}
}
