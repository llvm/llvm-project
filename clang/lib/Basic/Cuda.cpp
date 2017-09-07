#include "clang/Basic/Cuda.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang {

const char *CudaVersionToString(CudaVersion V) {
  switch (V) {
  case CudaVersion::UNKNOWN:
    return "unknown";
  case CudaVersion::CUDA_70:
    return "7.0";
  case CudaVersion::CUDA_75:
    return "7.5";
  case CudaVersion::CUDA_80:
    return "8.0";
  case CudaVersion::CUDA_90:
    return "9.0";
  }
  llvm_unreachable("invalid enum");
}

const char *CudaArchToString(CudaArch A) {
  switch (A) {
  case CudaArch::UNKNOWN:
    return "unknown";
  case CudaArch::SM_20:
    return "sm_20";
  case CudaArch::SM_21:
    return "sm_21";
  case CudaArch::SM_30:
    return "sm_30";
  case CudaArch::SM_32:
    return "sm_32";
  case CudaArch::SM_35:
    return "sm_35";
  case CudaArch::SM_37:
    return "sm_37";
  case CudaArch::SM_50:
    return "sm_50";
  case CudaArch::SM_52:
    return "sm_52";
  case CudaArch::SM_53:
    return "sm_53";
  case CudaArch::SM_60:
    return "sm_60";
  case CudaArch::SM_61:
    return "sm_61";
  case CudaArch::SM_62:
    return "sm_62";
  case CudaArch::SM_70:
    return "sm_70";
  }
  llvm_unreachable("invalid enum");
}

CudaArch StringToCudaArch(llvm::StringRef S) {
  return llvm::StringSwitch<CudaArch>(S)
      .Case("sm_20", CudaArch::SM_20)
      .Case("sm_21", CudaArch::SM_21)
      .Case("sm_30", CudaArch::SM_30)
      .Case("sm_32", CudaArch::SM_32)
      .Case("sm_35", CudaArch::SM_35)
      .Case("sm_37", CudaArch::SM_37)
      .Case("sm_50", CudaArch::SM_50)
      .Case("sm_52", CudaArch::SM_52)
      .Case("sm_53", CudaArch::SM_53)
      .Case("sm_60", CudaArch::SM_60)
      .Case("sm_61", CudaArch::SM_61)
      .Case("sm_62", CudaArch::SM_62)
      .Case("sm_70", CudaArch::SM_70)
      .Default(CudaArch::UNKNOWN);
}

const char *CudaVirtualArchToString(CudaVirtualArch A) {
  switch (A) {
  case CudaVirtualArch::UNKNOWN:
    return "unknown";
  case CudaVirtualArch::COMPUTE_20:
    return "compute_20";
  case CudaVirtualArch::COMPUTE_30:
    return "compute_30";
  case CudaVirtualArch::COMPUTE_32:
    return "compute_32";
  case CudaVirtualArch::COMPUTE_35:
    return "compute_35";
  case CudaVirtualArch::COMPUTE_37:
    return "compute_37";
  case CudaVirtualArch::COMPUTE_50:
    return "compute_50";
  case CudaVirtualArch::COMPUTE_52:
    return "compute_52";
  case CudaVirtualArch::COMPUTE_53:
    return "compute_53";
  case CudaVirtualArch::COMPUTE_60:
    return "compute_60";
  case CudaVirtualArch::COMPUTE_61:
    return "compute_61";
  case CudaVirtualArch::COMPUTE_62:
    return "compute_62";
  case CudaVirtualArch::COMPUTE_70:
    return "compute_70";
  }
  llvm_unreachable("invalid enum");
}

CudaVirtualArch StringToCudaVirtualArch(llvm::StringRef S) {
  return llvm::StringSwitch<CudaVirtualArch>(S)
      .Case("compute_20", CudaVirtualArch::COMPUTE_20)
      .Case("compute_30", CudaVirtualArch::COMPUTE_30)
      .Case("compute_32", CudaVirtualArch::COMPUTE_32)
      .Case("compute_35", CudaVirtualArch::COMPUTE_35)
      .Case("compute_37", CudaVirtualArch::COMPUTE_37)
      .Case("compute_50", CudaVirtualArch::COMPUTE_50)
      .Case("compute_52", CudaVirtualArch::COMPUTE_52)
      .Case("compute_53", CudaVirtualArch::COMPUTE_53)
      .Case("compute_60", CudaVirtualArch::COMPUTE_60)
      .Case("compute_61", CudaVirtualArch::COMPUTE_61)
      .Case("compute_62", CudaVirtualArch::COMPUTE_62)
      .Case("compute_70", CudaVirtualArch::COMPUTE_70)
      .Default(CudaVirtualArch::UNKNOWN);
}

CudaVirtualArch VirtualArchForCudaArch(CudaArch A) {
  switch (A) {
  case CudaArch::UNKNOWN:
    return CudaVirtualArch::UNKNOWN;
  case CudaArch::SM_20:
  case CudaArch::SM_21:
    return CudaVirtualArch::COMPUTE_20;
  case CudaArch::SM_30:
    return CudaVirtualArch::COMPUTE_30;
  case CudaArch::SM_32:
    return CudaVirtualArch::COMPUTE_32;
  case CudaArch::SM_35:
    return CudaVirtualArch::COMPUTE_35;
  case CudaArch::SM_37:
    return CudaVirtualArch::COMPUTE_37;
  case CudaArch::SM_50:
    return CudaVirtualArch::COMPUTE_50;
  case CudaArch::SM_52:
    return CudaVirtualArch::COMPUTE_52;
  case CudaArch::SM_53:
    return CudaVirtualArch::COMPUTE_53;
  case CudaArch::SM_60:
    return CudaVirtualArch::COMPUTE_60;
  case CudaArch::SM_61:
    return CudaVirtualArch::COMPUTE_61;
  case CudaArch::SM_62:
    return CudaVirtualArch::COMPUTE_62;
  case CudaArch::SM_70:
    return CudaVirtualArch::COMPUTE_70;
  }
  llvm_unreachable("invalid enum");
}

CudaVersion MinVersionForCudaArch(CudaArch A) {
  switch (A) {
  case CudaArch::UNKNOWN:
    return CudaVersion::UNKNOWN;
  case CudaArch::SM_20:
  case CudaArch::SM_21:
  case CudaArch::SM_30:
  case CudaArch::SM_32:
  case CudaArch::SM_35:
  case CudaArch::SM_37:
  case CudaArch::SM_50:
  case CudaArch::SM_52:
  case CudaArch::SM_53:
    return CudaVersion::CUDA_70;
  case CudaArch::SM_60:
  case CudaArch::SM_61:
  case CudaArch::SM_62:
    return CudaVersion::CUDA_80;
  case CudaArch::SM_70:
    return CudaVersion::CUDA_90;
  }
  llvm_unreachable("invalid enum");
}

} // namespace clang
