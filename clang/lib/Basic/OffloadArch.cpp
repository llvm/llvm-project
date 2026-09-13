//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "clang/Basic/OffloadArch.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/TargetParser/AMDGPUTargetParser.h"
#include "llvm/TargetParser/IntelGPUTargetParser.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {

OffloadArch OffloadArch::CudaDefault() {
  return getNVPTX(llvm::NVPTX::parseArch("sm_52"));
}

OffloadArch OffloadArch::HIPDefault() {
  return getAMDGPU(llvm::AMDGPU::parseArchAMDGCN("gfx906"));
}

const char *OffloadArchToString(OffloadArch A) {
  switch (A.targetArch()) {
  case OffloadArch::TargetArch::Unused:
    return "";
  case OffloadArch::TargetArch::Unknown:
    return "unknown";
  case OffloadArch::TargetArch::NVPTX:
    return llvm::NVPTX::getArchName(A.nvptxKind()).data();
  case OffloadArch::TargetArch::AMDGPU:
    return llvm::AMDGPU::getArchNameAMDGCN(A.amdgpuKind()).data();
  case OffloadArch::TargetArch::AMDGCNSPIRV:
    return "amdgcnspirv";
  case OffloadArch::TargetArch::IntelCPU:
    return "graniterapids";
  case OffloadArch::TargetArch::IntelGPU:
    return llvm::IntelGPU::getArchName(A.intelGPUKind()).data();
  case OffloadArch::TargetArch::Generic:
    return "generic";
  }
  return "unknown";
}

const char *OffloadArchToVirtualArchString(OffloadArch A) {
  switch (A.targetArch()) {
  case OffloadArch::TargetArch::NVPTX:
    return llvm::NVPTX::getVirtualArch(A.nvptxKind()).data();
  case OffloadArch::TargetArch::AMDGPU:
  case OffloadArch::TargetArch::AMDGCNSPIRV:
    return "compute_amdgcn";
  case OffloadArch::TargetArch::Unknown:
    return "unknown";
  case OffloadArch::TargetArch::Unused:
  case OffloadArch::TargetArch::IntelCPU:
  case OffloadArch::TargetArch::IntelGPU:
  case OffloadArch::TargetArch::Generic:
    return "";
  }
  return "unknown";
}

OffloadArch StringToOffloadArch(llvm::StringRef S) {
  // The empty string denotes the "unused" architecture.
  if (S.empty())
    return OffloadArch::getUnused();

  // Non-GPU-table pseudo/sentinel architectures.
  if (S == "amdgcnspirv")
    return OffloadArch::getAMDGCNSPIRV();
  if (S == "generic")
    return OffloadArch::getGeneric();
  // Intel CPU offload has a single processor and no TargetParser list.
  if (S == "graniterapids")
    return OffloadArch::getIntelCPU();

  // Otherwise defer to the vendor TargetParser GPU lists.
  if (llvm::NVPTX::GPUKind NV = llvm::NVPTX::parseArch(S))
    return OffloadArch::getNVPTX(NV);
  if (llvm::AMDGPU::GPUKind AK = llvm::AMDGPU::parseArchAMDGCN(S))
    return OffloadArch::getAMDGPU(AK);
  if (llvm::IntelGPU::GPUKind IK = llvm::IntelGPU::parseArch(S))
    return OffloadArch::getIntelGPU(IK);
  return OffloadArch::getUnknown();
}

void fillValidOffloadArchList(llvm::SmallVectorImpl<llvm::StringRef> &Values) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  Values.push_back(NAME);
#include "llvm/TargetParser/NVPTXTargetParser.def"
  llvm::AMDGPU::fillValidArchListAMDGCN(Values, llvm::Triple::NoSubArch);
  llvm::IntelGPU::fillValidArchList(Values);
}

OffloadArch getSubArchOffloadArch(llvm::Triple::SubArchType SubArch) {
  llvm::AMDGPU::GPUKind AK = llvm::AMDGPU::getGPUKindFromSubArch(SubArch);
  if (AK == llvm::AMDGPU::GK_NONE)
    return OffloadArch::getUnknown();
  return OffloadArch::getAMDGPU(AK);
}

llvm::Triple::SubArchType getOffloadArchSubArch(OffloadArch ID) {
  if (!ID.isAMDGPU())
    return llvm::Triple::NoSubArch;
  return llvm::AMDGPU::getSubArch(ID.amdgpuKind());
}

llvm::Triple OffloadArchToTriple(const llvm::Triple &DefaultToolchainTriple,
                                 OffloadArch ID) {
  if (ID.isAMDGCNSPIRV())
    return llvm::Triple(llvm::Triple::spirv64, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  if (ID.isNVPTX()) {
    llvm::Triple::ArchType Arch = DefaultToolchainTriple.isArch64Bit()
                                      ? llvm::Triple::nvptx64
                                      : llvm::Triple::nvptx;
    return llvm::Triple(Arch, llvm::Triple::NoSubArch, llvm::Triple::NVIDIA,
                        llvm::Triple::CUDA);
  }

  if (ID.isAMDGPU())
    return llvm::Triple(llvm::Triple::amdgpu, llvm::Triple::NoSubArch,
                        llvm::Triple::AMD, llvm::Triple::AMDHSA);

  return {};
}

} // namespace clang
