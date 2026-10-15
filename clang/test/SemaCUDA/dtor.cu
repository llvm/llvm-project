// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify=host
// RUN: %clang_cc1 %s -std=c++20 -fcuda-is-device -fsyntax-only -verify=dev

// host-no-diagnostics

#include "Inputs/cuda.h"

// Explicit __device__ virtual dtor ~B() reached from device code via
// destruction of a local variable should be walked by the deferred diag
// visitor and reach the host_fun() call in the dtor chain.
namespace ExplicitInstantiationExplicitDevDtor {
void host_fun() // dev-note {{'host_fun' declared here}}
{}

template <unsigned>
constexpr void hd_fun() { // dev-note {{in HD-promoted function 'hd_fun<8U>'}}
  host_fun(); // dev-error {{reference to __host__ function 'host_fun' in __host__ __device__ function}}
}

struct A {
  constexpr ~A() { // dev-note {{called by '~B'}}
     hd_fun<8>(); // dev-note {{called by '~A'}}
  }
};

template <typename T>
struct B {
public:
  virtual __device__ ~B() = default; // dev-note {{called by 'foo'}}
  A _a;
};

template class B<float>;
__device__ void foo() {
  B<float> x;
}
}

// Implicit H+D virtual dtor ~B() of an explicit instantiation that is
// not used from device code should not be eagerly walked by the deferred
// diag visitor. The host-only chain reachable from ~B() through ~A() is
// only relevant if device code actually constructs/destroys B<float>.
// C<float> is used from device foo() but its dtor chain is trivial.
namespace ExplicitInstantiationDtorNoAttr {
void host_fun()
{}

template <unsigned>
constexpr void hd_fun() {
  host_fun();
}

struct A {
  constexpr ~A() {
     hd_fun<8>();
  }
};

template <typename T>
struct B {
public:
  virtual ~B() = default;
  A _a;
};

template <typename T>
struct C {
public:
  virtual ~C() = default;
};

template class B<float>;
template class C<float>;
__device__ void foo() {
  C<float> x;
}
}

// Dtors of implicit template class instantiation are not
// conservatively inferred because the invalid usage can
// be diagnosed.
namespace ImplicitInstantiation {
void host_fun() // dev-note {{'host_fun' declared here}}
{}

template <unsigned>
constexpr void hd_fun() { // dev-note {{in HD-promoted function 'hd_fun<8U>'}}
  host_fun(); // dev-error {{reference to __host__ function 'host_fun' in __host__ __device__ function}}
}

struct A {
  constexpr ~A() { // dev-note {{called by '~B'}}
     hd_fun<8>(); // dev-note {{called by '~A'}}
  }
};

template <typename T>
struct B {
public:
  ~B() = default; // dev-note {{called by 'foo'}}
  A _a;
};

__device__ void foo() {
  B<float> x;
}
}
