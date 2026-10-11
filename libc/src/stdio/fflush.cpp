//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the fflush function which returns the specific
/// implementation based on CMake flags.
///
//===----------------------------------------------------------------------===//

#include "src/stdio/fflush.h"
#include "hdr/types/FILE.h"

#if defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/stdio/gpu/fflush.cpp"
#elif defined(LIBC_TARGET_OS_IS_BAREMETAL)
#include "src/stdio/baremetal/fflush.cpp"
#else
#include "src/stdio/generic/fflush.cpp"
#endif

namespace LIBC_NAMESPACE_DECL {

int fflush(::FILE *stream) {
#if defined(LIBC_TARGET_ARCH_IS_GPU)
  return GPU_FFLUSH::fflush(stream);
#elif defined(LIBC_TARGET_OS_IS_BAREMETAL)
  return BAREMETAL_FFLUSH::fflush(stream);
#else
  return GENERIC_FFLUSH::fflush(stream);
#endif
}

} // namespace LIBC_NAMESPACE_DECL
