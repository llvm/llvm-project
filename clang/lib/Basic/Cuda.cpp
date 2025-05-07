#include "clang/Basic/Cuda.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/VersionTuple.h"

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
  if (A == OffloadArch::UNKNOWN)
    return CudaVersion::UNKNOWN;

  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::CUDA_70;

  switch (A) {
  case OffloadArch::SM_20:
  case OffloadArch::SM_21:
  case OffloadArch::SM_30:
  case OffloadArch::SM_32_:
  case OffloadArch::SM_35:
  case OffloadArch::SM_37:
  case OffloadArch::SM_50:
  case OffloadArch::SM_52:
  case OffloadArch::SM_53:
    return CudaVersion::CUDA_70;
  case OffloadArch::SM_60:
  case OffloadArch::SM_61:
  case OffloadArch::SM_62:
    return CudaVersion::CUDA_80;
  case OffloadArch::SM_70:
    return CudaVersion::CUDA_90;
  case OffloadArch::SM_72:
    return CudaVersion::CUDA_91;
  case OffloadArch::SM_75:
    return CudaVersion::CUDA_100;
  case OffloadArch::SM_80:
    return CudaVersion::CUDA_110;
  case OffloadArch::SM_86:
    return CudaVersion::CUDA_111;
  case OffloadArch::SM_87:
    return CudaVersion::CUDA_114;
  case OffloadArch::SM_89:
  case OffloadArch::SM_90:
    return CudaVersion::CUDA_118;
  case OffloadArch::SM_90a:
    return CudaVersion::CUDA_120;
  case OffloadArch::SM_100:
  case OffloadArch::SM_100a:
  case OffloadArch::SM_101:
  case OffloadArch::SM_101a:
  case OffloadArch::SM_120:
  case OffloadArch::SM_120a:
    return CudaVersion::CUDA_128;
  default:
    llvm_unreachable("invalid enum");
  }
}

CudaVersion MaxVersionForOffloadArch(OffloadArch A) {
  // AMD GPUs do not depend on CUDA versions.
  if (IsAMDOffloadArch(A))
    return CudaVersion::NEW;

  switch (A) {
  case OffloadArch::UNKNOWN:
    return CudaVersion::UNKNOWN;
  case OffloadArch::SM_20:
  case OffloadArch::SM_21:
    return CudaVersion::CUDA_80;
  case OffloadArch::SM_30:
  case OffloadArch::SM_32_:
    return CudaVersion::CUDA_102;
  case OffloadArch::SM_35:
  case OffloadArch::SM_37:
    return CudaVersion::CUDA_118;
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
} // namespace clang
