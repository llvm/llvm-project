#include "clang/Basic/Cuda.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/TargetParser/NVPTXTargetParser.h"

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
  if (A == OffloadArch::Unknown)
    return CudaVersion::UNKNOWN;

  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::CUDA_70;

  switch (A) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case OffloadArch::KIND:                                                      \
    return CudaVersion::MIN_VER;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  default:
    llvm_unreachable("invalid enum");
  }
}

CudaVersion MaxVersionForOffloadArch(OffloadArch A) {
  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::NEW;

  switch (A) {
  case OffloadArch::Unknown:
    return CudaVersion::UNKNOWN;
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case OffloadArch::KIND:                                                      \
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
  switch (Arch) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case OffloadArch::KIND:                                                      \
    return SM_ID;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  default:
    break;
  }
  llvm_unreachable("invalid NVIDIA GPU architecture");
}

static llvm::NVPTX::GPUKind OffloadArchToNVPTXKind(OffloadArch Arch) {
  switch (Arch) {
#define NVPTX_GPU(NAME, KIND, VIRTUAL, SM_ID, MIN_VER, MAX_VER, SUFFIX)        \
  case OffloadArch::KIND:                                                      \
    return llvm::NVPTX::GK_##KIND;
#include "llvm/TargetParser/NVPTXTargetParser.def"
  default:
    return llvm::NVPTX::GK_NONE;
  }
}

bool IsNVIDIAAcceleratedOffloadArch(OffloadArch Arch) {
  return llvm::NVPTX::isAcceleratedArch(OffloadArchToNVPTXKind(Arch));
}

bool IsNVIDIAFamilySpecificOffloadArch(OffloadArch Arch) {
  return llvm::NVPTX::isFamilySpecificArch(OffloadArchToNVPTXKind(Arch));
}
} // namespace clang
