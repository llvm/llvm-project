// Verify the behavior of the denormal-fp-mode attributes in the way that
// rocm-device-libs should be built with. The bitcode should be compiled with
// denormal-fp-math-f32=dynamic, and should be replaced with the denormal mode
// of the final TU.

// Build the fake device library in the way rocm-device-libs should be built.
//
// RUN: %clang_cc1 -x cl -triple amdgcn-amd-amdhsa -fdenormal-fp-math-f32=dynamic \
// RUN:   -mcode-object-version=none -emit-llvm-bc \
// RUN:   %S/Inputs/ocml-sample.cl -o %t.dynamic.f32.bc
//
// RUN: %clang_cc1 -x cl -triple amdgcn-amd-amdhsa -fdenormal-fp-math=dynamic \
// RUN:   -mcode-object-version=none -emit-llvm-bc \
// RUN:   %S/Inputs/ocml-sample.cl -o %t.dynamic.full.bc



// Check the default behavior with no denormal-fp-math arguments.
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 -fcuda-is-device \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc \
// RUN:   -emit-llvm %s -o - | FileCheck -implicit-check-not=denormal-fp-math %s --check-prefixes=CHECK,INTERNALIZE


// Check an explicit full ieee request
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 -fcuda-is-device \
// RUN:    -fdenormal-fp-math=ieee \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc \
// RUN:   -emit-llvm %s -o - | FileCheck -implicit-check-not=denormal-fp-math %s --check-prefixes=CHECK,INTERNALIZE


// Check explicit f32-only flushing request
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,INTERNALIZE,IEEEF64-PSZF32


// Check explicit flush all request. Only the f32 component of the library is
// dynamic, so the linked functions should use IEEE as the base mode and the new
// functions preserve-sign.
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math=preserve-sign \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,INTERNALIZE,PSZ


// Check explicit f32-only, ieee-other flushing request
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math=ieee -fdenormal-fp-math-f32=preserve-sign \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,INTERNALIZE,IEEEF64-PSZF32


// Check inverse of normal usage. Requesting IEEE f32, with flushed f16/f64
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=ieee \
// RUN:   -mlink-builtin-bitcode %t.dynamic.f32.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,INTERNALIZE,IEEEF32-PSZF64-DYNF32


// Check backwards from the normal usage where both library components can be
// overridden.
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=ieee \
// RUN:   -mlink-builtin-bitcode %t.dynamic.full.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,INTERNALIZE,IEEEF32-PSZF64-DYNFULL



// Check the case where no internalization is performed
// RUN: %clang_cc1 -x hip -triple amdgcn-amd-amdhsa -target-cpu gfx803 \
// RUN:   -fcuda-is-device -fdenormal-fp-math=preserve-sign -fdenormal-fp-math-f32=ieee \
// RUN:   -mlink-bitcode-file %t.dynamic.full.bc -emit-llvm %s -o - \
// RUN: | FileCheck -implicit-check-not=denormal-fp-math --enable-var-scope %s --check-prefixes=CHECK,NOINTERNALIZE,NOINTERNALIZE-IEEEF32-PSZF64-DYNFULL



#define __device__ __attribute__((device))
#define __global__ __attribute__((global))

typedef _Float16 half;

extern "C" {
__device__ half do_f16_stuff(half a, half  b, half c);
__device__ float do_f32_stuff(float a, float b, float c);

// Currently all library functions are internalized. Check a weak function in
// case we ever choose to not internalize these. In that case, the safest thing
// to do would likely be to preserve the dynamic denormal-fp-math.
__attribute__((weak)) __device__ float weak_do_f32_stuff(float a, float b, float c);
__device__ double do_f64_stuff(double a, double b, double c);


  // CHECK: kernel_f16({{.*}}) #[[$KERNELATTR:[0-9]+]]
__global__ void kernel_f16(float* out, float* a, float* b, float* c) {
  int id = 0;
  out[id] = do_f16_stuff(a[id], b[id], c[id]);
}

// CHECK: kernel_f32({{.*}}) #[[$KERNELATTR]]
__global__ void kernel_f32(float* out, float* a, float* b, float* c) {
  int id = 0;
  out[id] = do_f32_stuff(a[id], b[id], c[id]);
  out[id] += weak_do_f32_stuff(a[id], b[id], c[id]);
}

// CHECK: kernel_f64({{.*}}) #[[$KERNELATTR]]
__global__ void kernel_f64(double* out, double* a, double* b, double* c) {
  int id = 0;
  out[id] = do_f64_stuff(a[id], b[id], c[id]);
}
}

// INTERNALIZE: define internal {{(noundef )?}}half @do_f16_stuff({{.*}}) #[[$FUNCATTR:[0-9]+]]
// INTERNALIZE: define internal {{(noundef )?}}float @do_f32_stuff({{.*}}) #[[$FUNCATTR]]
// INTERNALIZE: define internal {{(noundef )?}}double @do_f64_stuff({{.*}}) #[[$FUNCATTR]]
// INTERNALIZE: define internal {{(noundef )?}}float @weak_do_f32_stuff({{.*}}) #[[$WEAK_FUNCATTR:[0-9]+]]


// NOINTERNALIZE: define dso_local {{(noundef )?}}half @do_f16_stuff({{.*}}) #[[$FUNCATTR:[0-9]+]]
// NOINTERNALIZE: define dso_local {{(noundef )?}}float @do_f32_stuff({{.*}}) #[[$FUNCATTR]]
// NOINTERNALIZE: define dso_local {{(noundef )?}}double @do_f64_stuff({{.*}}) #[[$FUNCATTR]]
// NOINTERNALIZE: define weak {{(noundef )?}}float @weak_do_f32_stuff({{.*}}) #[[$WEAK_FUNCATTR:[0-9]+]]



// We should not be littering call sites with the attribute
// Everything should use the default ieee with no explicit attribute

// FIXME: Should check-not "denormal-fp-math" within the denormal-fp-math-f32
// lines.

// Default mode relies on the implicit check-not for the denormal-fp-math.

// PSZ: #[[$KERNELATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign"
// PSZ-SAME: "target-cpu"="gfx803"
// PSZ: #[[$FUNCATTR]] = { {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// PSZ-SAME: "target-cpu"="gfx803"
// PSZ: #[[$WEAK_FUNCATTR]] = { {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// PSZ-SAME: "target-cpu"="gfx803"

// FIXME: Should check-not "denormal-fp-math" within the line
// IEEEF64-PSZF32: #[[$KERNELATTR]] = { {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// IEEEF64-PSZF32-SAME: "target-cpu"="gfx803"
// IEEEF64-PSZF32: #[[$FUNCATTR]] = { {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// IEEEF64-PSZF32-SAME: "target-cpu"="gfx803"
// IEEEF64-PSZF32: #[[$WEAK_FUNCATTR]] = { {{.*}} "denormal-fp-math-f32"="preserve-sign,preserve-sign"
// IEEEF64-PSZF32-SAME: "target-cpu"="gfx803"

// IEEEF32-PSZF64-DYNF32: #[[$KERNELATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" {{.*}} "target-cpu"="gfx803" {{.*}}  }
// implicit check-not
// implicit check-not


// IEEEF32-PSZF64-DYNFULL: #[[$KERNELATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee"
// IEEEF32-PSZF64-DYNFULL-SAME: "target-cpu"="gfx803"
// IEEEF32-PSZF64-DYNFULL: #[[$FUNCATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee"
// IEEEF32-PSZF64-DYNFULL-SAME: "target-cpu"="gfx803"
// IEEEF32-PSZF64-DYNFULL: #[[$WEAK_FUNCATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee"
// IEEEF32-PSZF64-DYNFULL-SAME: "target-cpu"="gfx803"

// -mlink-bitcode-file doesn't internalize or propagate attributes.
// NOINTERNALIZE-IEEEF32-PSZF64-DYNFULL: #[[$KERNELATTR]] = { {{.*}} "denormal-fp-math"="preserve-sign,preserve-sign" "denormal-fp-math-f32"="ieee,ieee" {{.*}} "target-cpu"="gfx803" {{.*}} }
// NOINTERNALIZE-IEEEF32-PSZF64-DYNFULL: #[[$FUNCATTR]] = { {{.*}} "denormal-fp-math"="dynamic,dynamic" {{.*}} }
// NOINTERNALIZE-IEEEF32-PSZF64-DYNFULL: #[[$WEAK_FUNCATTR]] = { {{.*}} "denormal-fp-math"="dynamic,dynamic" {{.*}} }
