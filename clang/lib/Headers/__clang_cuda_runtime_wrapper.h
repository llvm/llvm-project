/*===---- __clang_cuda_runtime_wrapper.h - CUDA runtime support -------------===
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 *
 * CUDA headers are implemented in a way that currently makes it
 * impossible for user code to #include directly when compiling with
 * Clang. They present different view of CUDA-supplied functions
 * depending on where in NVCC's compilation pipeline the headers are
 * included. Neither of these modes provides function definitions with
 * correct attributes, so we use preprocessor to force the headers
 * into a form that Clang can use.
 *
 * Similarly to NVCC which -include's cuda_runtime.h, Clang -include's
 * this file during every CUDA compilation.
 */

#ifndef __CLANG_CUDA_RUNTIME_WRAPPER_H__
#define __CLANG_CUDA_RUNTIME_WRAPPER_H__

#if defined(__CUDA__) && defined(__clang__)

// Include some standard headers to avoid CUDA headers including them
// while some required macros (like __THROW) are in a weird state.
#include <stdlib.h>
#include <cmath>

// Preserve common macros that will be changed below by us or by CUDA
// headers.
#pragma push_macro("__THROW")
#pragma push_macro("__CUDA_ARCH__")

// WARNING: Preprocessor hacks below are based on specific details of
// CUDA-7.x headers and are not expected to work with any other
// version of CUDA headers.
#include "cuda.h"
#if !defined(CUDA_VERSION)
#error "cuda.h did not define CUDA_VERSION"
#elif CUDA_VERSION < 7000 || CUDA_VERSION > 7050
#error "Unsupported CUDA version!"
#endif

// Make largest subset of device functions available during host
// compilation -- SM_35 for the time being.
#ifndef __CUDA_ARCH__
#define __CUDA_ARCH__ 350
#endif

#include "cuda_builtin_vars.h"

// No need for device_launch_parameters.h as cuda_builtin_vars.h above
// has taken care of builtin variables declared in the file.
#define __DEVICE_LAUNCH_PARAMETERS_H__

// {math,device}_functions.h only have declarations of the
// functions. We don't need them as we're going to pull in their
// definitions from .hpp files.
#define __DEVICE_FUNCTIONS_H__
#define __MATH_FUNCTIONS_H__

#undef __CUDACC__
#define __CUDABE__
// Disables definitions of device-side runtime support stubs in
// cuda_device_runtime_api.h
#define __CUDADEVRT_INTERNAL__
#include "host_config.h"
#include "host_defines.h"
#include "driver_types.h"
#include "common_functions.h"
#undef __CUDADEVRT_INTERNAL__

#undef __CUDABE__
#define __CUDACC__
#include "cuda_runtime.h"

#undef __CUDACC__
#define __CUDABE__

// CUDA headers use __nvvm_memcpy and __nvvm_memset which Clang does
// not have at the moment. Emulate them with a builtin memcpy/memset.
#define __nvvm_memcpy(s,d,n,a) __builtin_memcpy(s,d,n)
#define __nvvm_memset(d,c,n,a) __builtin_memset(d,c,n)

#include "crt/host_runtime.h"
#include "crt/device_runtime.h"
// device_runtime.h defines __cxa_* macros that will conflict with
// cxxabi.h.
// FIXME: redefine these as __device__ functions.
#undef __cxa_vec_ctor
#undef __cxa_vec_cctor
#undef __cxa_vec_dtor
#undef __cxa_vec_new2
#undef __cxa_vec_new3
#undef __cxa_vec_delete2
#undef __cxa_vec_delete
#undef __cxa_vec_delete3
#undef __cxa_pure_virtual

// We need decls for functions in CUDA's libdevice with __device__
// attribute only. Alas they come either as __host__ __device__ or
// with no attributes at all. To work around that, define __CUDA_RTC__
// which produces HD variant and undef __host__ which gives us desided
// decls with __device__ attribute.
#pragma push_macro("__host__")
#define __host__
#define __CUDACC_RTC__
#include "device_functions_decls.h"
#undef __CUDACC_RTC__

// Temporarily poison __host__ macro to ensure it's not used by any of
// the headers we're about to include.
#define __host__ UNEXPECTED_HOST_ATTRIBUTE

// device_functions.hpp and math_functions*.hpp use 'static
// __forceinline__' (with no __device__) for definitions of device
// functions. Temporarily redefine __forceinline__ to include
// __device__.
#pragma push_macro("__forceinline__")
#define __forceinline__ __device__ __inline__ __attribute__((always_inline))
#include "device_functions.hpp"
#include "math_functions.hpp"
#include "math_functions_dbl_ptx3.hpp"
#pragma pop_macro("__forceinline__")

// Pull in host-only functions that are only available when neither
// __CUDACC__ nor __CUDABE__ are defined.
#undef __MATH_FUNCTIONS_HPP__
#undef __CUDABE__
#include "math_functions.hpp"
// Alas, additional overloads for these functions are hard to get to.
// Considering that we only need these overloads for a few functions,
// we can provide them here.
static inline float rsqrt(float a) { return rsqrtf(a); }
static inline float rcbrt(float a) { return rcbrtf(a); }
static inline float sinpi(float a) { return sinpif(a); }
static inline float cospi(float a) { return cospif(a); }
static inline void sincospi(float a, float *b, float *c) {
  return sincospi(a, b, c);
}
static inline float erfcinv(float a) { return erfcinvf(a); }
static inline float normcdfinv(float a) { return normcdfinvf(a); }
static inline float normcdf(float a) { return normcdff(a); }
static inline float erfcx(float a) { return erfcxf(a); }

// For some reason single-argument variant is not always declared by
// CUDA headers. Alas, device_functions.hpp included below needs it.
static inline __device__ void __brkpt(int c) { __brkpt(); }

// Now include *.hpp with definitions of various GPU functions.  Alas,
// a lot of thins get declared/defined with __host__ attribute which
// we don't want and we have to define it out. We also have to include
// {device,math}_functions.hpp again in order to extract the other
// branch of #if/else inside.

#define __host__
#undef __CUDABE__
#define __CUDACC__
#undef __DEVICE_FUNCTIONS_HPP__
#include "device_functions.hpp"
#include "device_atomic_functions.hpp"
#include "sm_20_atomic_functions.hpp"
#include "sm_32_atomic_functions.hpp"
#include "sm_20_intrinsics.hpp"
// sm_30_intrinsics.h has declarations that use default argument, so
// we have to include it and it will in turn include .hpp
#include "sm_30_intrinsics.h"
#include "sm_32_intrinsics.hpp"
#undef __MATH_FUNCTIONS_HPP__
#include "math_functions.hpp"
#pragma pop_macro("__host__")

#include "texture_indirect_functions.h"

// Restore state of __CUDA_ARCH__ and __THROW we had on entry.
#pragma pop_macro("__CUDA_ARCH__")
#pragma pop_macro("__THROW")

// Set up compiler macros expected to be seen during compilation.
#undef __CUDABE__
#define __CUDACC__
#define __NVCC__

#if defined(__CUDA_ARCH__)
// We need to emit IR declaration for non-existing __nvvm_reflect() to
// let backend know that it should be treated as const nothrow
// function which is what NVVMReflect pass expects to see.
extern "C" __device__ __attribute__((const)) int __nvvm_reflect(const void *);
static __device__ __attribute__((used)) int __nvvm_reflect_anchor() {
  return __nvvm_reflect("NONE");
}

// The nvptx vprintf syscall.  This doesn't actually need to be declared, but we
// declare it so that if someone else declares it with a different signature,
// we'll throw an error.
extern "C" __device__ int vprintf(const char*, const char*);
#endif

#endif // __CUDA__
#endif // __CLANG_CUDA_RUNTIME_WRAPPER_H__
