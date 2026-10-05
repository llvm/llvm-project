// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -fsyntax-only -verify=dev,dev-used %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -fsyntax-only -verify=host,host-used,host-key %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -femit-all-decls -fsyntax-only -verify=dev,dev-all %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fcxx-exceptions \
// RUN:   -femit-all-decls -fsyntax-only -verify=host,host-all,host-key %s
// RUN: %clang_cc1 -std=c++20 -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -fsyntax-only -verify=dev,dev-used %s
// RUN: %clang_cc1 -std=c++20 -x hip -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fsyntax-only -verify=host,host-used,host-key %s

// Defaulted functions are immediate-escalating in C++20, which defers their
// emission status, so check them in C++17 too.
// RUN: %clang_cc1 -std=c++17 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -femit-all-decls -fsyntax-only -verify=dev,dev-all,dev-cxx17 %s

// An inline function cannot be a key function in this ABI.
// RUN: %clang_cc1 -std=c++20 -triple arm64-apple-macosx -fcxx-exceptions \
// RUN:   -fsyntax-only -verify=host,host-used %s

// Implicit host device templates are only emitted for the device if they are
// used there, even with -femit-all-decls.
// RUN: %clang_cc1 -std=c++20 -x hip -triple amdgcn-amd-amdhsa \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -foffload-implicit-host-device-templates -femit-all-decls \
// RUN:   -fsyntax-only -verify=dev,dev-all %s

// The errors are reported in a compilation that emits code too, before the
// functions are emitted.
// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -fcxx-exceptions -fcuda-is-device \
// RUN:   -emit-llvm -o /dev/null -verify=dev,dev-used %s

// The deferred diagnostics of a __host__ __device__ function are reported if
// the function is emitted. Besides the functions that are used, CodeGen emits
// functions it is forced to emit, whether or not they are used.

#include "Inputs/cuda.h"

__device__ void device_only(); // #device_only
// host-note@#device_only 4 {{'device_only' declared here}}
// host-all-note@#device_only 2 {{'device_only' declared here}}
// host-key-note@#device_only {{'device_only' declared here}}
__host__ void host_only(); // #host_only

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

// An inline key function is emitted with the vtable.
struct KeyFunction {
  virtual __host__ __device__ void key();
};
inline __host__ __device__ void KeyFunction::key() {
  throw NULL;
  // dev-error@-1 {{cannot use 'throw' in __host__ __device__ function}}
  device_only();
  // host-key-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
}

#if __cplusplus >= 202002L
// dev-all-note@#host_only {{'host_only' declared here}}

// Immediate functions are never emitted, even if forced.
__attribute__((used)) consteval __host__ __device__ int immediate(int x) {
  if (x) {
    (void)&device_only;
    throw NULL;
  }
  return 0;
}

// Neither is a function that becomes immediate by calling an immediate
// function, which is only known once its body is complete.
consteval int id(int x) { return x; }
template <class T> constexpr __host__ __device__ int escalating(T x) {
  if (x) {
    device_only();
    host_only();
  }
  return id(x);
}
int escalated = escalating(0);

// Unlike a function that does not become immediate.
template <class T> constexpr __host__ __device__ int not_escalating(T x) {
  if (x) {
    device_only();
    // host-all-error@-1 {{reference to __device__ function 'device_only' in __host__ __device__ function}}
    host_only();
    // dev-all-error@-1 {{reference to __host__ function 'host_only' in __host__ __device__ function}}
  }
  return x;
}
int not_escalated = not_escalating(0);
#endif

// Lambdas are only emitted if they are used.
inline auto lambda = [] __attribute__((used)) __host__ __device__ {
  throw NULL;
  device_only();
};

// With -foffload-implicit-host-device-templates, this is an implicit host
// device function only used on the host.
template <class T> T implicit_hd(T x) {
  host_only();
  return x;
}
int host_user() { return implicit_hd(0); }

// An available externally definition is only emitted to be inlined into its
// callers.
extern inline __attribute__((gnu_inline, used)) __host__ __device__ void
available_externally() {
  throw NULL;
  device_only();
}

// Implicit functions and functions defaulted on their first declaration are
// only emitted when used, here only by host functions.
struct HostOnly {
  __host__ HostOnly() {}
  __host__ ~HostOnly() {} // #host_only_dtor
  // dev-all-note@#host_only_dtor {{'~HostOnly' declared here}}
};
struct Defaulted {
  HostOnly m;
  __host__ __device__ ~Defaulted() = default;
};
void use_defaulted() { Defaulted d; }
struct Base {
  __host__ __device__ Base(int) {}
};
struct Inheriting : Base {
  using Base::Base;
  HostOnly m;
};
void use_inheriting() { Inheriting i(0); }

// Unlike a function defaulted after its first declaration.
struct DefaultedOutOfLine {
  HostOnly m;
  __host__ __device__ ~DefaultedOutOfLine();
};
inline __host__ __device__ DefaultedOutOfLine::~DefaultedOutOfLine() = default;
// dev-all-error@-1 {{reference to __host__ function '~HostOnly' in __host__ __device__ function}}
// dev-cxx17-note@-2 {{in defaulted destructor for 'DefaultedOutOfLine' first required here}}
