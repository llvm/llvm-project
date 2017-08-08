// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s

#include "Inputs/cuda.h"

//------------------------------------------------------------------------------
// Test 1: infer default ctor to be host.

struct A1_with_host_ctor {
  A1_with_host_ctor() {}
};

// The implicit default constructor is inferred to be host because it only needs
// to invoke a single host constructor (A1_with_host_ctor's). So we'll encounter
// an error when calling it from a __device__ function, but not from a __host__
// function.
struct B1_with_implicit_default_ctor : A1_with_host_ctor {
};

// expected-note@-3 {{call to __host__ function from __device__}}
// expected-note@-4 {{candidate constructor (the implicit copy constructor) not viable}}
// expected-note@-5 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo() {
  B1_with_implicit_default_ctor b;
}

__device__ void devicefoo() {
  B1_with_implicit_default_ctor b; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 2: infer default ctor to be device.

struct A2_with_device_ctor {
  __device__ A2_with_device_ctor() {}
};

struct B2_with_implicit_default_ctor : A2_with_device_ctor {
};

// expected-note@-3 {{call to __device__ function from __host__}}
// expected-note@-4 {{candidate constructor (the implicit copy constructor) not viable}}
// expected-note@-5 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo2() {
  B2_with_implicit_default_ctor b;  // expected-error {{no matching constructor}}
}

__device__ void devicefoo2() {
  B2_with_implicit_default_ctor b;
}

//------------------------------------------------------------------------------
// Test 3: infer copy ctor

struct A3_with_device_ctors {
  __host__ A3_with_device_ctors() {}
  __device__ A3_with_device_ctors(const A3_with_device_ctors&) {}
};

struct B3_with_implicit_ctors : A3_with_device_ctors {
};
// expected-note@-2 2{{call to __device__ function from __host__ function}}
// expected-note@-3 {{default constructor}}


void hostfoo3() {
  B3_with_implicit_ctors b;  // this is OK because the inferred default ctor
                             // here is __host__
  B3_with_implicit_ctors b2 = b; // expected-error {{no matching constructor}}

}

//------------------------------------------------------------------------------
// Test 4: infer default ctor from a field, not a base

struct A4_with_host_ctor {
  A4_with_host_ctor() {}
};

struct B4_with_implicit_default_ctor {
  A4_with_host_ctor field;
};

// expected-note@-4 {{call to __host__ function from __device__}}
// expected-note@-5 {{candidate constructor (the implicit copy constructor) not viable}}
// expected-note@-6 {{candidate constructor (the implicit move constructor) not viable}}

void hostfoo4() {
  B4_with_implicit_default_ctor b;
}

__device__ void devicefoo4() {
  B4_with_implicit_default_ctor b; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 5: copy ctor with non-const param

struct A5_copy_ctor_constness {
  __host__ A5_copy_ctor_constness() {}
  __host__ A5_copy_ctor_constness(A5_copy_ctor_constness&) {}
};

struct B5_copy_ctor_constness : A5_copy_ctor_constness {
};

// expected-note@-3 {{candidate constructor (the implicit copy constructor) not viable: call to __host__ function from __device__ function}}
// expected-note@-4 {{candidate constructor (the implicit default constructor) not viable}}

void hostfoo5(B5_copy_ctor_constness& b_arg) {
  B5_copy_ctor_constness b = b_arg;
}

__device__ void devicefoo5(B5_copy_ctor_constness& b_arg) {
  B5_copy_ctor_constness b = b_arg; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 6: explicitly defaulted ctor: since they are spelled out, they have
// a host/device designation explicitly so no inference needs to be done.

struct A6_with_device_ctor {
  __device__ A6_with_device_ctor() {}
};

struct B6_with_defaulted_ctor : A6_with_device_ctor {
  __host__ B6_with_defaulted_ctor() = default;
};

// expected-note@-3 {{candidate constructor not viable: call to __host__ function from __device__ function}}
// expected-note@-5 {{candidate constructor (the implicit copy constructor) not viable}}
// expected-note@-6 {{candidate constructor (the implicit move constructor) not viable}}

__device__ void devicefoo6() {
  B6_with_defaulted_ctor b; // expected-error {{no matching constructor}}
}

//------------------------------------------------------------------------------
// Test 7: copy assignment operator

struct A7_with_copy_assign {
  A7_with_copy_assign() {}
  __device__ A7_with_copy_assign& operator=(const A7_with_copy_assign&) {}
};

struct B7_with_copy_assign : A7_with_copy_assign {
};

// expected-note@-3 {{candidate function (the implicit copy assignment operator) not viable: call to __device__ function from __host__ function}}
// expected-note@-4 {{candidate function (the implicit move assignment operator) not viable: call to __device__ function from __host__ function}}

void hostfoo7() {
  B7_with_copy_assign b1, b2;
  b1 = b2; // expected-error {{no viable overloaded '='}}
}

//------------------------------------------------------------------------------
// Test 8: move assignment operator

// definitions for std::move
namespace std {
inline namespace foo {
template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T&> { typedef T type; };
template <class T> struct remove_reference<T&&> { typedef T type; };

template <class T> typename remove_reference<T>::type&& move(T&& t);
}
}

struct A8_with_move_assign {
  A8_with_move_assign() {}
  __device__ A8_with_move_assign& operator=(A8_with_move_assign&&) {}
  __device__ A8_with_move_assign& operator=(const A8_with_move_assign&) {}
};

struct B8_with_move_assign : A8_with_move_assign {
};

// expected-note@-3 {{candidate function (the implicit copy assignment operator) not viable: call to __device__ function from __host__ function}}
// expected-note@-4 {{candidate function (the implicit move assignment operator) not viable: call to __device__ function from __host__ function}}

void hostfoo8() {
  B8_with_move_assign b1, b2;
  b1 = std::move(b2); // expected-error {{no viable overloaded '='}}
}
