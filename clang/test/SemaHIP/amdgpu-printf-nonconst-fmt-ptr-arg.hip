// REQUIRES: amdgpu-registered-target, spirv-registered-target, nvptx-registered-target, x86-registered-target
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -Wno-format-security -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv64-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -Wno-format-security -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip \
// RUN:    -Wno-format-security -fsyntax-only -verify=none %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -x hip -fcuda-is-device \
// RUN:    -Wno-format-security -fsyntax-only -verify=none %s
// RUN: %clang_cc1 -triple amdgpu9.0a-amd-amdhsa -x hip -fcuda-is-device \
// RUN:    -fno-builtin-printf -Wno-format-security -fsyntax-only -verify=nobuiltin %s

// none-no-diagnostics

// The AMDGPU printf lowering can only find %s specifiers in a format string
// known at compile time, so a pointer argument requires such a format.

#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __global__ __attribute__((global))

extern "C" __host__ __device__ int printf(const char *format, ...);

__device__ void NonConstFmtPtrArgs(const char *fmt, const char *s, int *p) {
  printf(fmt, s); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
  printf(fmt, 1, p); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
  printf(fmt, nullptr); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
  // -fno-builtin-printf only affects printf, not __builtin_printf.
  __builtin_printf(fmt, s); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}} \
                            // nobuiltin-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
}

__device__ void NonConstFmtNoPtrArgs(const char *fmt) {
  printf(fmt);
  printf(fmt, 1, 2.0, 'c');
}

// An unevaluated printf is never called, so it is not lowered.
__device__ void UnevaluatedNonConstFmtPtrArgs(const char *fmt, const char *s) {
  (void)sizeof(printf(fmt, s));
  using T = decltype(printf(fmt, s));
  (void)noexcept(printf(fmt, s));
}

template <typename T>
__device__ void UnevaluatedTypeDependentFmt(T fmt, const char *s) {
  (void)sizeof(printf(fmt, s));
}

__device__ const char DeviceFmt[] = "%s\n";
__device__ constexpr char ConstexprFmt[] = "x%s\n";

__device__ void ConstFmtPtrArgs(const char *s, int *p) {
  constexpr const char LocalFmt[] = "%s\n";
  printf("%s %p\n", s, p);
  printf(LocalFmt, s);
  printf(DeviceFmt, s);
  printf(ConstexprFmt + 1, s);
}

__host__ void HostOnly(const char *fmt, const char *s) {
  printf(fmt, s);
}

inline __host__ __device__ void HostDeviceUnused(const char *fmt, const char *s) {
  printf(fmt, s);
}

inline __host__ __device__ void HostDeviceUsed(const char *fmt, const char *s) {
  printf(fmt, s); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
}

__global__ void Kernel(const char *fmt, const char *s) {
  HostDeviceUsed(fmt, s); // expected-note {{called by 'Kernel'}}
}

template <typename T> __device__ void TypeDependentFmt(T fmt, const char *s) {
  printf(fmt, s); // expected-error {{AMDGPU printf requires a constant format string when pointer arguments are passed}}
}

template <const char *Fmt> __device__ void ValueDependentFmt(const char *s) {
  printf(Fmt, s);
}

__device__ void Instantiate(const char *fmt, const char *s) {
  TypeDependentFmt(fmt, s); // expected-note {{in instantiation of function template specialization 'TypeDependentFmt<const char *>' requested here}}
  ValueDependentFmt<ConstexprFmt>(s);
  UnevaluatedTypeDependentFmt(fmt, s);
}
