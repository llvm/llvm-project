//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file handles the resolution of the specific fflush function based on
/// the build type
///
//===----------------------------------------------------------------------===//

#if defined(LIBC_TARGET_ARCH_IS_GPU)
#include "./gpu/fflush.cpp"
#elif defined(LIBC_TARGET_OS_IS_BAREMETAL)
#include "./baremetal/fflush.cpp"
#else
#include "./generic/fflush.cpp"
#endif

namespace LIBC_NAMESPACE_DECL {

// fflush handles the resolution of function call
// based on CMAKE build args, allowing the callers
// of fflush to be bothered about different implementations
// for different architectures (cpu, gpu or baremetal)
int fflush(::FILE *stream) {
#if defined(LIBC_TARGET_ARCH_IS_GPU)
  return ::FflushGPU::fflush(stream);
#elif defined(LIBC_TARGET_OS_IS_BAREMETAL)
  return ::FflushBaremetal::fflush(stream);
#else
  return ::FflushGeneric::fflush(stream);
#endif
}
} // namespace LIBC_NAMESPACE_DECL
