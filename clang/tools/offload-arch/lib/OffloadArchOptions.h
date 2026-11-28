//===- OffloadArchOptions.h - offload-arch settings ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_TOOLS_OFFLOAD_ARCH_LIB_OFFLOADARCHOPTIONS_H
#define CLANG_TOOLS_OFFLOAD_ARCH_LIB_OFFLOADARCHOPTIONS_H

enum class HipApiVersion {
  Auto,       // Automatic fallback (R0600 -> R0000 -> unversioned)
  R0600,      // Force R0600 API (HIP 6.x+)
  R0000,      // Force R0000 API (legacy HIP)
  Unversioned // Force unversioned API (very old HIP)
};

// Settings shared by the offload-arch tool and the GPU detection code.
extern bool Verbose;
extern HipApiVersion HipApi;

#endif
