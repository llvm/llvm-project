// RUN: %clang_cc1 -triple amdgcn -fcuda-is-device -std=c++11 -fvisibility hidden -fapply-global-visibility-to-externs \
// RUN:     -emit-llvm -o - -x hip %s -fsyntax-only -verify
// RUN: %clang_cc1 -triple x86_64 -std=c++11 \
// RUN:     -emit-llvm -o - -x hip %s -fsyntax-only -verify

#define __device__ __attribute__((device))
#define __constant__ __attribute__((constant))
#define __hip_pinned_shadow__ __attribute((hip_pinned_shadow))

struct textureReference {
  int a;
};

template <class T, int texType, int hipTextureReadMode>
struct texture : public textureReference {
texture() { a = 1; }
};

__hip_pinned_shadow__ texture<float, 2, 1> tex;
__device__ __hip_pinned_shadow__ texture<float, 2, 1> tex2; // expected-error{{'hip_pinned_shadow' and 'device' attributes are not compatible}}
                                                            // expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables}}
                                                            // expected-note@-2{{conflicting attribute is here}}
__constant__ __hip_pinned_shadow__ texture<float, 2, 1> tex3; // expected-error{{'hip_pinned_shadow' and 'constant' attributes are not compatible}}
                                                              // expected-error@-1{{dynamic initialization is not supported for __device__, __constant__, and __shared__ variables}}
                                                              // expected-note@-2{{conflicting attribute is here}}
