// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++20 \
// RUN:   -fsyntax-only -verify %s
// expected-no-diagnostics

// An explicit class template instantiation with an implicit
// __host__ __device__ virtual destructor must not produce device
// diagnostics for host-only callees reachable through its destructor
// chain when no device code references the class. The destructor is a
// candidate for device emission only because of the explicit
// instantiation, not because of any device use, so deferred device
// diagnostics should not be raised against its body.

#include "Inputs/cuda.h"

// A host-only function reachable from an implicit H+D destructor body.
void host_only_dealloc() {}

// constexpr functions get implicit __host__ __device__, but their bodies
// can still call host-only functions on the runtime path.
template <unsigned long>
constexpr void deallocate() {
  host_only_dealloc();
}

struct alloc_holder {
  constexpr ~alloc_holder() { deallocate<8>(); }
};

template <typename T>
struct Base {
  virtual ~Base() = default;
};

template <typename T>
struct Derived : Base<T> {
  alloc_holder m_data;
};

template class Derived<double>;
