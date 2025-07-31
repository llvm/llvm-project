//===--- Cuda.h - Utilities for compiling CUDA code  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CUDA_H
#define LLVM_CLANG_BASIC_CUDA_H

#include "clang/Basic/OffloadArch.h"

namespace llvm {
class StringRef;
class Twine;
class VersionTuple;
} // namespace llvm

namespace clang {

enum class CudaVersion {
  UNKNOWN,
  CUDA_70,
  CUDA_75,
  CUDA_80,
  CUDA_90,
  CUDA_91,
  CUDA_92,
  CUDA_100,
  CUDA_101,
  CUDA_102,
  CUDA_110,
  CUDA_111,
  CUDA_112,
  CUDA_113,
  CUDA_114,
  CUDA_115,
  CUDA_116,
  CUDA_117,
  CUDA_118,
  CUDA_120,
  CUDA_121,
  CUDA_122,
  CUDA_123,
  CUDA_124,
  CUDA_125,
  CUDA_126,
  CUDA_128,
  FULLY_SUPPORTED = CUDA_123,
  PARTIALLY_SUPPORTED =
      CUDA_128, // Partially supported. Proceed with a warning.
  NEW = 10000,  // Too new. Issue a warning, but allow using it.
};
const char *CudaVersionToString(CudaVersion V);
// Input is "Major.Minor"
CudaVersion CudaStringToVersion(const llvm::Twine &S);

enum class CUDAFunctionTarget {
  Device,
  Global,
  Host,
  HostDevice,
  InvalidTarget
};

/// Get the earliest CudaVersion that supports the given OffloadArch.
CudaVersion MinVersionForOffloadArch(OffloadArch A);

/// Get the latest CudaVersion that supports the given OffloadArch.
CudaVersion MaxVersionForOffloadArch(OffloadArch A);

//  Various SDK-dependent features that affect CUDA compilation
enum class CudaFeature {
  // CUDA-9.2+ uses a new API for launching kernels.
  CUDA_USES_NEW_LAUNCH,
  // CUDA-10.1+ needs explicit end of GPU binary registration.
  CUDA_USES_FATBIN_REGISTER_END,
};

CudaVersion ToCudaVersion(llvm::VersionTuple);
bool CudaFeatureEnabled(llvm::VersionTuple, CudaFeature);
bool CudaFeatureEnabled(CudaVersion, CudaFeature);

} // namespace clang

#endif
