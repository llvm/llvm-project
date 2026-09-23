#include "cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:            -fclangir -I%S/../CodeGenCUDA/Inputs/ -emit-cir %s -o - \
// RUN:   | FileCheck --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:            -fclangir -I%S/../CodeGenCUDA/Inputs/ -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fhip-new-launch-api \
// RUN:            -I%S/../CodeGenCUDA/Inputs/ -emit-llvm %s -o - \
// RUN:   | FileCheck --check-prefix=LLVM %s

// The host-side kernel handle takes the linkage and visibility of the kernel's
// device stub. Kernels with internal linkage must get an internal handle in
// every TU that uses them; an external one causes multiple-definition link
// errors.

template <class T> static __global__ void static_tmpl(T *p) {}
template <class T> __global__ void tmpl(T *p) {}
static __global__ void static_kernel(int *p) {}
namespace {
__global__ void anon_kernel(int *p) {}
}
__global__ void ext_kernel(int *p) {}
__attribute__((visibility("hidden"))) __global__ void hidden_kernel(int *p) {}

// Explicit instantiation definition: weak_odr, in a comdat.
template <class T> __global__ void inst(T *p) {}
template __global__ void inst<float>(float *p);

// Explicit specialization of a template that is only declared: strong
// external, no comdat.
template <class T> __global__ void decl_only(T *p);
template <> __global__ void decl_only<int>(int *p) {}

// Referenced before its definition, so the handle is created before the
// stub's linkage is known.
static __global__ void fwd_kernel(int *p);

void launch(int *p) {
  static_tmpl<int><<<1, 1>>>(p);
  tmpl<int><<<1, 1>>>(p);
  static_kernel<<<1, 1>>>(p);
  anon_kernel<<<1, 1>>>(p);
  ext_kernel<<<1, 1>>>(p);
  hidden_kernel<<<1, 1>>>(p);
  decl_only<int><<<1, 1>>>(p);
  fwd_kernel<<<1, 1>>>(p);
}

static __global__ void fwd_kernel(int *p) {}

// CIR-DAG: cir.global constant external @_Z10ext_kernelPi = #cir.global_view<@_Z25__device_stub__ext_kernelPi>
// CIR-DAG: cir.global hidden constant external @_Z13hidden_kernelPi = #cir.global_view<@_Z28__device_stub__hidden_kernelPi>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL11static_tmplIiEvPT_ = #cir.global_view<@_ZL26__device_stub__static_tmplIiEvPT_>
// CIR-DAG: cir.global constant linkonce_odr comdat @_Z4tmplIiEvPT_ = #cir.global_view<@_Z19__device_stub__tmplIiEvPT_>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL13static_kernelPi = #cir.global_view<@_ZL28__device_stub__static_kernelPi>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZN12_GLOBAL__N_111anon_kernelEPi = #cir.global_view<@_ZN12_GLOBAL__N_126__device_stub__anon_kernelEPi>
// CIR-DAG: cir.global "private" constant internal dso_local @_ZL10fwd_kernelPi = #cir.global_view<@_ZL25__device_stub__fwd_kernelPi>
// CIR-DAG: cir.global constant weak_odr comdat @_Z4instIfEvPT_ = #cir.global_view<@_Z19__device_stub__instIfEvPT_>
// CIR-DAG: cir.global constant external @_Z9decl_onlyIiEvPT_ = #cir.global_view<@_Z24__device_stub__decl_onlyIiEvPT_>

// LLVM-DAG: @_Z10ext_kernelPi = constant ptr @_Z25__device_stub__ext_kernelPi, align 8
// LLVM-DAG: @_Z13hidden_kernelPi = hidden constant ptr @_Z28__device_stub__hidden_kernelPi, align 8
// LLVM-DAG: @_ZL11static_tmplIiEvPT_ = internal constant ptr @_ZL26__device_stub__static_tmplIiEvPT_, align 8
// LLVM-DAG: @_ZL13static_kernelPi = internal constant ptr @_ZL28__device_stub__static_kernelPi, align 8
// LLVM-DAG: @_ZN12_GLOBAL__N_111anon_kernelEPi = internal constant ptr @_ZN12_GLOBAL__N_126__device_stub__anon_kernelEPi, align 8
// LLVM-DAG: @_ZL10fwd_kernelPi = internal constant ptr @_ZL25__device_stub__fwd_kernelPi, align 8
// LLVM-DAG: @_Z4tmplIiEvPT_ = linkonce_odr constant ptr @_Z19__device_stub__tmplIiEvPT_, comdat, align 8
// LLVM-DAG: @_Z4instIfEvPT_ = weak_odr constant ptr @_Z19__device_stub__instIfEvPT_, comdat, align 8
// LLVM-DAG: @_Z9decl_onlyIiEvPT_ = constant ptr @_Z24__device_stub__decl_onlyIiEvPT_, align 8
