//===--- SYCL.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SYCL_H
#define LLVM_CLANG_BASIC_SYCL_H

#include "clang/Basic/Cuda.h"

namespace llvm {
class StringRef;
template <unsigned InternalLen> class SmallString;
} // namespace llvm

namespace clang {
// List of architectures (Intel CPUs and Intel GPUs)
// that support SYCL offloading.
enum class SYCLSupportedIntelArchs {
  // Intel CPUs
  UNKNOWN,
  SKYLAKEAVX512,
  COREAVX2,
  COREI7AVX,
  COREI7,
  WESTMERE,
  SANDYBRIDGE,
  IVYBRIDGE,
  BROADWELL,
  COFFEELAKE,
  ALDERLAKE,
  SKYLAKE,
  SKX,
  CASCADELAKE,
  ICELAKECLIENT,
  ICELAKESERVER,
  SAPPHIRERAPIDS,
  GRANITERAPIDS,
  // Intel GPUs
  BDW,
  SKL,
  KBL,
  CFL,
  APL,
  BXT,
  GLK,
  WHL,
  AML,
  CML,
  ICLLP,
  ICL,
  EHL,
  JSL,
  TGLLP,
  TGL,
  RKL,
  ADL_S,
  RPL_S,
  ADL_P,
  ADL_N,
  DG1,
  ACM_G10,
  DG2_G10,
  ACM_G11,
  DG2_G11,
  ACM_G12,
  DG2_G12,
  PVC,
  PVC_VG,
  MTL_U,
  MTL_S,
  ARL_U,
  ARL_S,
  MTL_H,
  ARL_H,
  BMG_G21,
  LNL_M,
};

// Check if the given Arch value is a Generic AMD GPU.
// Currently GFX*_GENERIC AMD GPUs do not support SYCL offloading.
// This list is used to filter out GFX*_GENERIC AMD GPUs in
// `IsSYCLSupportedAMDGPUArch`.
static inline bool IsAMDGenericGPUArch(OffloadArch Arch) {
  return Arch == OffloadArch::GFX9_GENERIC ||
         Arch == OffloadArch::GFX10_1_GENERIC ||
         Arch == OffloadArch::GFX10_3_GENERIC ||
         Arch == OffloadArch::GFX11_GENERIC ||
         Arch == OffloadArch::GFX12_GENERIC;
}

// Check if the given Arch value is a valid SYCL supported AMD GPU.
static inline bool IsSYCLSupportedAMDGPUArch(OffloadArch Arch) {
  return Arch >= OffloadArch::GFX700 && Arch < OffloadArch::AMDGCNSPIRV &&
         !IsAMDGenericGPUArch(Arch);
}

// Check if the given Arch value is a valid SYCL supported NVidia GPU.
static inline bool IsSYCLSupportedNVidiaGPUArch(OffloadArch Arch) {
  return Arch >= OffloadArch::SM_50 && Arch <= OffloadArch::SM_90a;
}

// Check if the given Arch value is a valid SYCL supported Intel CPU.
static inline bool IsSYCLSupportedIntelCPUArch(SYCLSupportedIntelArchs Arch) {
  return Arch >= SYCLSupportedIntelArchs::SKYLAKEAVX512 &&
         Arch <= SYCLSupportedIntelArchs::GRANITERAPIDS;
}

// Check if the given Arch value is a valid SYCL supported Intel GPU.
static inline bool IsSYCLSupportedIntelGPUArch(SYCLSupportedIntelArchs Arch) {
  return Arch >= SYCLSupportedIntelArchs::BDW &&
         Arch <= SYCLSupportedIntelArchs::LNL_M;
}

// Check if the user provided value for --offload-arch is a valid
// SYCL supported Intel AOT target.
SYCLSupportedIntelArchs
StringToOffloadArchSYCL(llvm::StringRef ArchNameAsString);

// This is a mapping between the user provided --offload-arch value for Intel
// GPU targets and the spir64_gen device name accepted by OCLOC (the Intel GPU
// AOT compiler).
llvm::StringRef mapIntelGPUArchName(llvm::StringRef ArchName);
llvm::SmallString<64> getGenDeviceMacro(llvm::StringRef DeviceName);

} // namespace clang

#endif // LLVM_CLANG_BASIC_SYCL_H
