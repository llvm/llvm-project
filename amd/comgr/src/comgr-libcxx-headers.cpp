//===-- comgr-libcxx-headers.cpp - Embedded libc++ headers for HIPRTC ----===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "comgr-libcxx-headers.h"

namespace COMGR {

#ifdef COMGR_HAS_LIBCXX_HEADERS

// getLibcxxHeaderFiles() and getClangBuiltinHeaderFiles() are defined in
// the generated libcxx_headers.cpp (built as embed-libcxx-headers-lib).

bool hasEmbeddedLibcxxHeaders() {
  return !getLibcxxHeaderFiles().empty();
}

#else // !COMGR_HAS_LIBCXX_HEADERS

bool hasEmbeddedLibcxxHeaders() {
  return false;
}

llvm::ArrayRef<ResourceDirResource> getLibcxxHeaderFiles() {
  return {};
}

llvm::ArrayRef<ResourceDirResource> getClangBuiltinHeaderFiles() {
  return {};
}

#endif // COMGR_HAS_LIBCXX_HEADERS

} // namespace COMGR
