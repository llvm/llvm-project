//===-- comgr-libcxx-headers.h - Embedded libc++ headers for HIPRTC ------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Embedded libc++ and Clang builtin headers for HIPRTC, using the same
/// ResourceDirResource infrastructure as the clang resource directory.
///
//===----------------------------------------------------------------------===//

#ifndef COMGR_LIBCXX_HEADERS_H
#define COMGR_LIBCXX_HEADERS_H

#include "comgr-resource-directory.h"

namespace COMGR {

bool hasEmbeddedLibcxxHeaders();

/// Paths relative to <install>/include/c++/v1/.
llvm::ArrayRef<ResourceDirResource> getLibcxxHeaderFiles();

/// Paths relative to <resource-dir>/include/.
llvm::ArrayRef<ResourceDirResource> getClangBuiltinHeaderFiles();

} // namespace COMGR

#endif // COMGR_LIBCXX_HEADERS_H
