#include "clang/Basic/Cuda.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"
#include <cassert>

namespace clang {

struct CudaVersionMapEntry {
  const char *Name;
  CudaVersion Version;
  llvm::VersionTuple TVersion;
};
#define CUDA_ENTRY(major, minor)                                               \
  {                                                                            \
    #major "." #minor, CudaVersion::CUDA_##major##minor,                       \
        llvm::VersionTuple(major, minor)                                       \
  }

static const CudaVersionMapEntry CudaNameVersionMap[] = {
    CUDA_ENTRY(7, 0),
    CUDA_ENTRY(7, 5),
    CUDA_ENTRY(8, 0),
    CUDA_ENTRY(9, 0),
    CUDA_ENTRY(9, 1),
    CUDA_ENTRY(9, 2),
    CUDA_ENTRY(10, 0),
    CUDA_ENTRY(10, 1),
    CUDA_ENTRY(10, 2),
    CUDA_ENTRY(11, 0),
    CUDA_ENTRY(11, 1),
    CUDA_ENTRY(11, 2),
    CUDA_ENTRY(11, 3),
    CUDA_ENTRY(11, 4),
    CUDA_ENTRY(11, 5),
    CUDA_ENTRY(11, 6),
    CUDA_ENTRY(11, 7),
    CUDA_ENTRY(11, 8),
    CUDA_ENTRY(12, 0),
    CUDA_ENTRY(12, 1),
    CUDA_ENTRY(12, 2),
    CUDA_ENTRY(12, 3),
    CUDA_ENTRY(12, 4),
    CUDA_ENTRY(12, 5),
    CUDA_ENTRY(12, 6),
    CUDA_ENTRY(12, 8),
    CUDA_ENTRY(12, 9),
    CUDA_ENTRY(13, 0),
    CUDA_ENTRY(13, 1),
    CUDA_ENTRY(13, 2),
    CUDA_ENTRY(13, 3),
    CUDA_ENTRY(13, 4),
    {"", CudaVersion::NEW, llvm::VersionTuple(std::numeric_limits<int>::max())},
    {"unknown", CudaVersion::UNKNOWN, {}} // End of list tombstone.
};
#undef CUDA_ENTRY

const char *CudaVersionToString(CudaVersion V) {
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->Version == V)
      return I->Name;

  return CudaVersionToString(CudaVersion::UNKNOWN);
}

CudaVersion CudaStringToVersion(const llvm::Twine &S) {
  std::string VS = S.str();
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->Name == VS)
      return I->Version;
  return CudaVersion::UNKNOWN;
}

CudaVersion ToCudaVersion(llvm::VersionTuple Version) {
  for (auto *I = CudaNameVersionMap; I->Version != CudaVersion::UNKNOWN; ++I)
    if (I->TVersion == Version)
      return I->Version;
  return CudaVersion::UNKNOWN;
}

CudaVersion MinVersionForOffloadArch(OffloadArch A) {
  if (A.isUnknown())
    return CudaVersion::UNKNOWN;

  // AMD GPUs do not depend on CUDA versions.
  if (A.isAMDGPU() || A.isSPIRV())
    return CudaVersion::CUDA_70;

  switch (A.nvptxKind()) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case llvm::NVPTX::GK_##KIND:                                                 \
    return CudaVersion::MIN_VER;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  default:
    llvm_unreachable("invalid enum");
  }
}

CudaVersion MaxVersionForOffloadArch(OffloadArch A) {
  // AMD GPUs do not depend on CUDA versions.
  if (A.isAMDGPU() || A.isSPIRV())
    return CudaVersion::NEW;

  if (!A.isNVPTX())
    return CudaVersion::UNKNOWN;

  switch (A.nvptxKind()) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case llvm::NVPTX::GK_##KIND:                                                 \
    return CudaVersion::MAX_VER;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  default:
    return CudaVersion::NEW;
  }
}

bool CudaFeatureEnabled(llvm::VersionTuple Version, CudaFeature Feature) {
  return CudaFeatureEnabled(ToCudaVersion(Version), Feature);
}

bool CudaFeatureEnabled(CudaVersion Version, CudaFeature Feature) {
  switch (Feature) {
  case CudaFeature::CUDA_USES_NEW_LAUNCH:
    return Version >= CudaVersion::CUDA_92;
  case CudaFeature::CUDA_USES_FATBIN_REGISTER_END:
    return Version >= CudaVersion::CUDA_101;
  }
  llvm_unreachable("Unknown CUDA feature.");
}

unsigned CudaArchToID(OffloadArch Arch) {
  assert(Arch.isNVPTX() && "invalid NVIDIA GPU architecture");
  return llvm::NVPTX::getSmVersion(Arch.nvptxKind());
}

bool IsNVIDIAAcceleratedOffloadArch(OffloadArch Arch) {
  return Arch.isNVPTX() && llvm::NVPTX::isAcceleratedArch(Arch.nvptxKind());
}

bool IsNVIDIAFamilySpecificOffloadArch(OffloadArch Arch) {
  return Arch.isNVPTX() && llvm::NVPTX::isFamilySpecificArch(Arch.nvptxKind());
}
} // namespace clang
