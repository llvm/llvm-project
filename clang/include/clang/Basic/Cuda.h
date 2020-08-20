//===--- Cuda.h - Utilities for compiling CUDA code  ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_CUDA_H
#define LLVM_CLANG_BASIC_CUDA_H

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
  LATEST = CUDA_110,
  LATEST_SUPPORTED = CUDA_101,
};
const char *CudaVersionToString(CudaVersion V);
// Input is "Major.Minor"
CudaVersion CudaStringToVersion(const llvm::Twine &S);

enum class CudaArch {
  UNKNOWN,
  SM_20,
  SM_21,
  SM_30,
  SM_32,
  SM_35,
  SM_37,
  SM_50,
  SM_52,
  SM_53,
  SM_60,
  SM_61,
  SM_62,
  SM_70,
  SM_72,
  SM_75,
  SM_80,
  GFX600,
  GFX601,
  GFX700,
  GFX701,
  GFX702,
  GFX703,
  GFX704,
  GFX801,
  GFX802,
  GFX803,
  GFX810,
  GFX900,
  GFX902,
  GFX904,
  GFX906,
  GFX908,
  GFX909,
  GFX1010,
  GFX1011,
  GFX1012,
  GFX1030,
  GFX1031,
  LAST,
};

static inline bool IsNVIDIAGpuArch(CudaArch A) {
  return A >= CudaArch::SM_20 && A < CudaArch::GFX600;
}

static inline bool IsAMDGpuArch(CudaArch A) {
  return A >= CudaArch::GFX600 && A < CudaArch::LAST;
}

const char *CudaArchToString(CudaArch A);
const char *CudaArchToVirtualArchString(CudaArch A);

// The input should have the form "sm_20".
CudaArch StringToCudaArch(llvm::StringRef S);

/// Get the earliest CudaVersion that supports the given CudaArch.
CudaVersion MinVersionForCudaArch(CudaArch A);

/// Get the latest CudaVersion that supports the given CudaArch.
CudaVersion MaxVersionForCudaArch(CudaArch A);

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
