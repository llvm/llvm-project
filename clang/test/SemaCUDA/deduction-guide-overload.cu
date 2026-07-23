// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -fcuda-is-device -verify %s
// RUN: %clang_cc1 -std=c++20 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:            -verify %s
// expected-no-diagnostics

#include "Inputs/cuda.h"

// This test exercises class template argument deduction (CTAD) when there are
// multiple constructors that differ only by constraints. In CUDA/HIP mode, the
// implementation must *not* collapse implicit deduction guides that have the
// same function type but different constraints; otherwise, CTAD can lose viable
// candidates.

template <typename T>
concept Signed = __is_signed(T);

template <typename T>
concept NotSigned = !Signed<T>;

// 1) Constrained ctors with different constraints: ensure we keep
// deduction guides that differ only by constraints.

template <typename T>
struct OverloadCTAD {
  __host__ __device__ OverloadCTAD(T) requires Signed<T>;
  __host__ __device__ OverloadCTAD(T) requires NotSigned<T>;
};

__host__ __device__ void use_overload_ctad_hd() {
  OverloadCTAD a(1);   // T = int, uses Signed-constrained guide
  OverloadCTAD b(1u);  // T = unsigned int, uses NotSigned-constrained guide
}

__device__ void use_overload_ctad_dev() {
  OverloadCTAD c(1);
  OverloadCTAD d(1u);
}

__global__ void use_overload_ctad_global() {
  OverloadCTAD e(1);
  OverloadCTAD f(1u);
}

// 2) Add a pair of constructors that have the same signature and the same
// constraint but differ only by CUDA target attributes. This exercises the
// case where two implicit deduction guides would be identical except for
// their originating constructor's CUDA target.

template <typename T>
struct OverloadCTADTargets {
  __host__ OverloadCTADTargets(T) requires Signed<T>;
  __device__ OverloadCTADTargets(T) requires Signed<T>;
};

__host__ void use_overload_ctad_targets_host() {
  OverloadCTADTargets g(1);
}

__device__ void use_overload_ctad_targets_device() {
  OverloadCTADTargets h(1);
}

// 3) Unconstrained host/device duplicates: identical signatures and no
// constraints, differing only by CUDA target attributes.

template <typename T>
struct UnconstrainedHD {
  __host__ UnconstrainedHD(T);
  __device__ UnconstrainedHD(T);
};

__host__ __device__ void use_unconstrained_hd_hd() {
  UnconstrainedHD u1(1);
}

__device__ void use_unconstrained_hd_dev() {
  UnconstrainedHD u2(1);
}

__global__ void use_unconstrained_hd_global() {
  UnconstrainedHD u3(1);
}

// 4) Constrained vs unconstrained ctors with the same signature: guides
// must not be collapsed away when constraints differ.

template <typename T>
concept IsInt = __is_same(T, int);

template <typename T>
struct ConstrainedVsUnconstrained {
  __host__ __device__ ConstrainedVsUnconstrained(T);
  __host__ __device__ ConstrainedVsUnconstrained(T) requires IsInt<T>;
};

__host__ __device__ void use_constrained_vs_unconstrained_hd() {
  ConstrainedVsUnconstrained a(1);    // T = int, constrained guide viable
  ConstrainedVsUnconstrained b(1u);   // T = unsigned, only unconstrained guide
}

__device__ void use_constrained_vs_unconstrained_dev() {
  ConstrainedVsUnconstrained c(1);
  ConstrainedVsUnconstrained d(1u);
}

__global__ void use_constrained_vs_unconstrained_global() {
  ConstrainedVsUnconstrained e(1);
  ConstrainedVsUnconstrained f(1u);
}

