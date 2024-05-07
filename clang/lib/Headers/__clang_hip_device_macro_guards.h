/*===---- __clang_hip_device_macro_guards.h - guards for HIP device macros -===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

/*
 * WARNING: This header is intended to be directly -include'd by
 * the compiler and is not supposed to be included by users.
 *
 */

#ifndef __CLANG_HIP_DEVICE_MACRO_GUARDS_H__
#define __CLANG_HIP_DEVICE_MACRO_GUARDS_H__

#if __HIP__
#if !defined(__HIP_DEVICE_COMPILE__)
// The __AMDGCN_WAVEFRONT_SIZE macros cannot hold meaningful values during host
// compilation as devices are not initialized when the macros are defined and
// there may indeed be devices with differing wavefront sizes in the same
// system. This code issues diagnostics when the macros are used in host code.

#undef __AMDGCN_WAVEFRONT_SIZE
#undef __AMDGCN_WAVEFRONT_SIZE__

// Reference __hip_device_macro_guard in a way that is legal in preprocessor
// directives and does not affect the value so that appropriate diagnostics are
// issued. Function calls, casts, or the comma operator would make the macro
// illegal for use in preprocessor directives.
#define __AMDGCN_WAVEFRONT_SIZE (!__hip_device_macro_guard ? 64 : 64)
#define __AMDGCN_WAVEFRONT_SIZE__ (!__hip_device_macro_guard ? 64 : 64)

// This function is referenced by the macro in device functions during host
// compilation, it SHOULD NOT cause a diagnostic.
__attribute__((device)) static constexpr int __hip_device_macro_guard(void) {
  return -1;
}

// This function is referenced by the macro in host functions during host
// compilation, it SHOULD cause a diagnostic.
__attribute__((
    host, deprecated("The __AMDGCN_WAVEFRONT_SIZE macros do not correspond "
                     "to the device(s) when used in host code and may only "
                     "be used in device code."))) static constexpr int
__hip_device_macro_guard(void) {
  return -1;
}
// TODO Change "deprecated" to "unavailable" to cause hard errors instead of
// warnings.
#endif
#endif // __HIP__
#endif // __CLANG_HIP_DEVICE_MACRO_GUARDS_H__
