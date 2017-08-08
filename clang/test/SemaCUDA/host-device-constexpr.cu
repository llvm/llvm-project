// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -isystem %S/Inputs %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -isystem %S/Inputs %s -fcuda-is-device

#include "Inputs/cuda.h"

// Declares one function and pulls it into namespace ns:
//
//   __device__ int OverloadMe();
//   namespace ns { using ::OverloadMe; }
//
// Clang cares that this is done in a system header.
#include <overload.h>

// Opaque type used to determine which overload we're invoking.
struct HostReturnTy {};

// These shouldn't become host+device because they already have attributes.
__host__ constexpr int HostOnly() { return 0; }
// expected-note@-1 0+ {{not viable}}
__device__ constexpr int DeviceOnly() { return 0; }
// expected-note@-1 0+ {{not viable}}

constexpr int HostDevice() { return 0; }

// This should be a host-only function, because there's a previous __device__
// overload in <overload.h>.
constexpr HostReturnTy OverloadMe() { return HostReturnTy(); }

namespace ns {
// The "using" statement in overload.h should prevent OverloadMe from being
// implicitly host+device.
constexpr HostReturnTy OverloadMe() { return HostReturnTy(); }
}  // namespace ns

// This is an error, because NonSysHdrOverload was not defined in a system
// header.
__device__ int NonSysHdrOverload() { return 0; }
// expected-note@-1 {{conflicting __device__ function declared here}}
constexpr int NonSysHdrOverload() { return 0; }
// expected-error@-1 {{constexpr function 'NonSysHdrOverload' without __host__ or __device__ attributes}}

// Variadic device functions are not allowed, so this is just treated as
// host-only.
constexpr void Variadic(const char*, ...);
// expected-note@-1 {{call to __host__ function from __device__ function}}

__host__ void HostFn() {
  HostOnly();
  DeviceOnly(); // expected-error {{no matching function}}
  HostReturnTy x = OverloadMe();
  HostReturnTy y = ns::OverloadMe();
  Variadic("abc", 42);
}

__device__ void DeviceFn() {
  HostOnly(); // expected-error {{no matching function}}
  DeviceOnly();
  int x = OverloadMe();
  int y = ns::OverloadMe();
  Variadic("abc", 42); // expected-error {{no matching function}}
}

__host__ __device__ void HostDeviceFn() {
#ifdef __CUDA_ARCH__
  int y = OverloadMe();
#else
  constexpr HostReturnTy y = OverloadMe();
#endif
}
