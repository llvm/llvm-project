// REQUIRES: amdgpu-registered-target
// REQUIRED: spirv-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple amdgcn -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple spirv64-amd-amdhsa -Wno-unused-value %s
// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64 -aux-triple spirv64-amd-amdhsa -Wno-unused-value %s

#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

__device__ void foo(__amdgpu_feature_predicate_t l) {
  decltype(__builtin_amdgcn_processor_is("gfx900")) what; // expected-error {{'what' has type __amdgpu_feature_predicate_t, which is not constructible}}
  typeof(__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep)) why; // expected-error {{'why' has type __amdgpu_feature_predicate_t, which is not constructible}}

  bool b = true;
  __amdgpu_feature_predicate_t v = false; // expected-error {{'v' has type __amdgpu_feature_predicate_t, which is not constructible}}
  static_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{static_cast from 'bool' to '__amdgpu_feature_predicate_t' is not allowed}}
  dynamic_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{invalid target type '__amdgpu_feature_predicate_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{reinterpret_cast from 'bool' to '__amdgpu_feature_predicate_t' is not allowed}}
  __amdgpu_feature_predicate_t k; // expected-error {{'k' has type __amdgpu_feature_predicate_t, which is not constructible}}
  int *ip = (int *)l; // expected-error {{cannot cast from type '__amdgpu_feature_predicate_t' to pointer type 'int *'}}
  void *vp = (void *)l; // expected-error {{cannot cast from type '__amdgpu_feature_predicate_t' to pointer type 'void *'}}
}

__global__ void bar(__amdgpu_feature_predicate_t l) {
  decltype(__builtin_amdgcn_processor_is("gfx900")) what; // expected-error {{'what' has type __amdgpu_feature_predicate_t, which is not constructible}}
  typeof(__builtin_amdgcn_is_invocable(__builtin_amdgcn_s_sleep)) why; // expected-error {{'why' has type __amdgpu_feature_predicate_t, which is not constructible}}

  bool b = true;
  __amdgpu_feature_predicate_t v = false; // expected-error {{'v' has type __amdgpu_feature_predicate_t, which is not constructible}}
  static_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{static_cast from 'bool' to '__amdgpu_feature_predicate_t' is not allowed}}
  dynamic_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{invalid target type '__amdgpu_feature_predicate_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__amdgpu_feature_predicate_t>(b); // expected-error {{reinterpret_cast from 'bool' to '__amdgpu_feature_predicate_t' is not allowed}}
  __amdgpu_feature_predicate_t k; // expected-error {{'k' has type __amdgpu_feature_predicate_t, which is not constructible}}
  int *ip = (int *)l; // expected-error {{cannot cast from type '__amdgpu_feature_predicate_t' to pointer type 'int *'}}
  void *vp = (void *)l; // expected-error {{cannot cast from type '__amdgpu_feature_predicate_t' to pointer type 'void *'}}
}

static_assert(sizeof(__amdgpu_feature_predicate_t) == 0); // expected-error {{invalid application of 'sizeof' to sizeless type '__amdgpu_feature_predicate_t'}}
static_assert(alignof(__amdgpu_feature_predicate_t) == 0); // expected-error {{invalid application of 'alignof' to sizeless type '__amdgpu_feature_predicate_t'}};
