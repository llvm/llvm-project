// RUN: %clang_cc1 -fcuda-is-device -fsyntax-only -verify=expected,dev %s
// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -x hip -fcuda-is-device -fsyntax-only -verify=expected,dev %s
// RUN: %clang_cc1 -x hip -fsyntax-only -verify %s

#include "Inputs/cuda.h"

namespace std {
class type_info {};
} // namespace std

struct B {
  __host__ __device__ virtual ~B() {}
};
struct D : B {};

void host(B *b) {
  (void)dynamic_cast<D *>(b);
  (void)typeid(*b);
}

__device__ void device(B *b, D *d) {
  (void)dynamic_cast<D *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  (void)dynamic_cast<D &>(*b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
  (void)typeid(D);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  (void)typeid(*b);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
  (void)sizeof(typeid(int));
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}

  // As with -fno-rtti, these don't use RTTI and are allowed.
  (void)dynamic_cast<void *>(b);
  (void)dynamic_cast<B *>(d);
}

__global__ void kernel(B *b) {
  (void)typeid(*b);
  // expected-error@-1 {{cannot use 'typeid' in __global__ function as RTTI is not available in device code}}
}

// Check that it's an error to use RTTI from a __host__ __device__ function if
// and only if it's codegen'ed for device.

__host__ __device__ void hd1(B *b) {
  (void)dynamic_cast<D *>(b);
  // dev-error@-1 {{cannot use 'dynamic_cast' in __host__ __device__ function as RTTI is not available in device code}}
}

// No error, never instantiated on device.
inline __host__ __device__ void hd2(B *b) { (void)typeid(*b); }
void call_hd2(B *b) { hd2(b); }

// Error, instantiated on device.
inline __host__ __device__ void hd3(B *b) {
  (void)typeid(*b);
  // dev-error@-1 {{cannot use 'typeid' in __host__ __device__ function as RTTI is not available in device code}}
}
__device__ void call_hd3(B *b) { hd3(b); }
// dev-note@-1 {{called by 'call_hd3'}}

// Templates are checked when they are instantiated.
template <class T> __device__ T *tmpl_unused(B *b) {
  return dynamic_cast<T *>(b);
}

template <class T> __device__ T *tmpl(B *b) {
  return dynamic_cast<T *>(b);
  // expected-error@-1 {{cannot use 'dynamic_cast' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl(B *b) { tmpl<D>(b); }
// expected-note@-1 {{in instantiation of function template specialization 'tmpl<D>' requested here}}

template <class T> __device__ void tmpl_typeid(T *t) {
  (void)typeid(*t);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl_typeid(B *b) { tmpl_typeid(b); }
// expected-note@-1 {{in instantiation of function template specialization 'tmpl_typeid<B>' requested here}}

template <class T> __device__ void tmpl_typeid_unused(T *t) {
  (void)typeid(*t);
}

// A non-dependent operand is checked once, in the template definition.
template <class T> __device__ void tmpl_nondependent() {
  (void)typeid(int);
  // expected-error@-1 {{cannot use 'typeid' in __device__ function as RTTI is not available in device code}}
}
__device__ void call_tmpl_nondependent() { tmpl_nondependent<int>(); }

// A host virtual function in a class that also has device virtual functions
// is not device code.
struct Fix {
  __device__ virtual void run() {}
  virtual D *init(B *b) { return dynamic_cast<D *>(b); }
};
__device__ void use_fix() { Fix f; f.run(); }
