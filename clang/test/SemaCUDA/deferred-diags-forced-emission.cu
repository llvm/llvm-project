// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -fsyntax-only -verify=dev,dev-used %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify=host,host-used %s
// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -femit-all-decls \
// RUN:   -fsyntax-only -verify=dev,dev-all %s
// RUN: %clang_cc1 -fcxx-exceptions -femit-all-decls -fsyntax-only \
// RUN:   -verify=host,host-all %s
// RUN: %clang_cc1 -x hip -fcxx-exceptions -fcuda-is-device -fsyntax-only \
// RUN:   -verify=dev,dev-used %s
// RUN: %clang_cc1 -x hip -fcxx-exceptions -fsyntax-only -verify=host,host-used %s

// The errors are reported in a compilation that emits code too, before the
// functions are emitted.
// RUN: %clang_cc1 -fcxx-exceptions -fcuda-is-device -emit-llvm -o /dev/null \
// RUN:   -verify=dev,dev-used %s

// The deferred diagnostics of a __host__ __device__ function are reported if
// the function is emitted. Besides the functions that are used, CodeGen emits
// functions it is forced to emit, whether or not they are used.

#include "Inputs/cuda.h"

__device__ void device_only(); // #device_only
// host-note@#device_only 4 {{'device_only' declared here}}
// host-all-note@#device_only {{'device_only' declared here}}

inline __host__ __device__ __attribute__((used)) void used_fn() {
  throw NULL;
  // dev-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
  device_only();
  // host-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}

inline __host__ __device__ __attribute__((constructor)) void ctor_fn() {
  // dev-error@-1 {{CUDA does not support global constructors for __device__ functions}}
  device_only();
  // host-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}

inline __host__ __device__ __attribute__((destructor)) void dtor_fn() {
  // dev-error@-1 {{CUDA does not support global destructors for __device__ functions}}
  device_only();
  // host-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}

// Not used, so not emitted unless all declarations are.
inline __host__ __device__ void unused_fn() {
  throw NULL;
  // dev-all-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
  device_only();
  // host-all-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}

// A function used by a forced function is emitted as well.
inline __host__ __device__ void callee() {
  throw NULL;
  // dev-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
  device_only();
  // host-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}
inline __host__ __device__ __attribute__((used)) void caller() { callee(); }
// Without -femit-all-decls, callee is only emitted as caller's callee.
// dev-used-note@-2 {{called by 'caller'}}
// host-used-note@-3 {{called by 'caller'}}

// The definition inherits the attribute from an earlier declaration.
__attribute__((used)) inline __host__ __device__ void inherited();
inline __host__ __device__ void inherited() {
  throw NULL;
  // dev-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
}

// Internal linkage instead of inline.
static __host__ __device__ __attribute__((used)) void internal() {
  throw NULL;
  // dev-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
}

// An uninstantiated template is not emitted, even with -femit-all-decls.
template <class T> inline __host__ __device__ void uninstantiated() {
  throw NULL;
  device_only();
}
