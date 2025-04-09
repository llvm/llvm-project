// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify=host
// RUN: %clang_cc1 %s -std=c++20 -fcuda-is-device -fsyntax-only -verify=dev

// host-no-diagnostics

#include "Inputs/cuda.h"

// Virtual dtor ~B() of explicit instantiation B<float> must
// be emitted, which causes host_fun() called.
namespace ExplicitInstantiationExplicitDevDtor {
void host_fun() // dev-note {{'host_fun' declared here}}
{}

template <unsigned>
constexpr void hd_fun() {
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
  virtual __device__ ~B() = default;
  A _a;
};

template class B<float>;
}

// The implicit host/device attrs of virtual dtor ~B() should be
// conservatively inferred, where constexpr member dtor's should
// not be considered device since they may call host functions.
// Therefore B<float>::~B() should not have implicit device attr.
// However C<float>::~C() should have implicit device attr since
// it is trivial.
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
constexpr void hd_fun() {
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
