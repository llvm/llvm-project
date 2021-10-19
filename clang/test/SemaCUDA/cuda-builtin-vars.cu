// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda" -fcuda-is-device -fsyntax-only -verify \
// RUN:   -verify-ignore-unexpected=note %s

#include "__clang_cuda_builtin_vars.h"
__attribute__((global))
void kernel(int *out) {
  int i = 0;
  out[i++] = threadIdx.x;
  threadIdx.x = 0; // expected-error {{no setter defined for property 'x'}}
  out[i++] = threadIdx.y;
  threadIdx.y = 0; // expected-error {{no setter defined for property 'y'}}
  out[i++] = threadIdx.z;
  threadIdx.z = 0; // expected-error {{no setter defined for property 'z'}}

  out[i++] = blockIdx.x;
  blockIdx.x = 0; // expected-error {{no setter defined for property 'x'}}
  out[i++] = blockIdx.y;
  blockIdx.y = 0; // expected-error {{no setter defined for property 'y'}}
  out[i++] = blockIdx.z;
  blockIdx.z = 0; // expected-error {{no setter defined for property 'z'}}

  out[i++] = blockDim.x;
  blockDim.x = 0; // expected-error {{no setter defined for property 'x'}}
  out[i++] = blockDim.y;
  blockDim.y = 0; // expected-error {{no setter defined for property 'y'}}
  out[i++] = blockDim.z;
  blockDim.z = 0; // expected-error {{no setter defined for property 'z'}}

  out[i++] = gridDim.x;
  gridDim.x = 0; // expected-error {{no setter defined for property 'x'}}
  out[i++] = gridDim.y;
  gridDim.y = 0; // expected-error {{no setter defined for property 'y'}}
  out[i++] = gridDim.z;
  gridDim.z = 0; // expected-error {{no setter defined for property 'z'}}

  out[i++] = warpSize;
  warpSize = 0; // expected-error {{cannot assign to variable 'warpSize' with const-qualified type 'const int'}}
  // expected-note@__clang_cuda_builtin_vars.h:* {{variable 'warpSize' declared const here}}

  // Make sure we can't construct or assign to the special variables.
  __cuda_builtin_threadIdx_t x; // expected-error {{call to deleted constructor of '__cuda_builtin_threadIdx_t'}}

  __cuda_builtin_threadIdx_t y = threadIdx; // expected-error {{call to deleted constructor of '__cuda_builtin_threadIdx_t'}}

  threadIdx = threadIdx; // expected-error {{overload resolution selected deleted operator '='}}

  void *ptr = &threadIdx; // expected-error {{overload resolution selected deleted operator '&'}}

  // Following line should've caused an error as one is not allowed to
  // take address of a built-in variable in CUDA. Alas there's no way
  // to prevent getting address of a 'const int', so the line
  // currently compiles without errors or warnings.
  const void *wsptr = &warpSize;
}
