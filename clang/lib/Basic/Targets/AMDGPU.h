//===--- AMDGPU.h - Declare AMDGPU target feature support -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares AMDGPU TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_AMDGPU_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_AMDGPU_H

#include "clang/AST/Type.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY AMDGPUTargetInfo final : public TargetInfo {

  static const Builtin::Info BuiltinInfo[];
  static const char *const GCCRegNames[];

  struct LLVM_LIBRARY_VISIBILITY AddrSpace {
    unsigned Generic, Global, Local, Constant, Private;
    AddrSpace(bool IsGenericZero_ = false) {
      if (IsGenericZero_) {
        Generic = 0;
        Global = 1;
        Local = 3;
        Constant = 2;
        Private = 5;
      } else {
        Generic = 4;
        Global = 1;
        Local = 3;
        Constant = 2;
        Private = 0;
      }
    }
  };

  /// \brief The GPU profiles supported by the AMDGPU target.
  enum GPUKind {
    GK_NONE,
    GK_R600,
    GK_R600_DOUBLE_OPS,
    GK_R700,
    GK_R700_DOUBLE_OPS,
    GK_EVERGREEN,
    GK_EVERGREEN_DOUBLE_OPS,
    GK_NORTHERN_ISLANDS,
    GK_CAYMAN,
    GK_GFX6,
    GK_GFX7,
    GK_GFX8,
    GK_GFX9
  } GPU;

  bool hasFP64 : 1;
  bool hasFMAF : 1;
  bool hasLDEXPF : 1;
  const AddrSpace AS;

  static bool hasFullSpeedFMAF32(StringRef GPUName) {
    return parseAMDGCNName(GPUName) >= GK_GFX9;
  }

  static bool isAMDGCN(const llvm::Triple &TT) {
    return TT.getArch() == llvm::Triple::amdgcn;
  }

  static bool isGenericZero(const llvm::Triple &TT) {
    return TT.getEnvironmentName() == "amdgiz" ||
           TT.getEnvironmentName() == "amdgizcl";
  }

public:
  AMDGPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void setAddressSpaceMap(bool DefaultIsPrivate);

  void adjust(LangOptions &Opts) override;

  uint64_t getPointerWidthV(unsigned AddrSpace) const override {
    if (GPU <= GK_CAYMAN)
      return 32;

    if (AddrSpace == AS.Private || AddrSpace == AS.Local) {
      return 32;
    }
    return 64;
  }

  uint64_t getPointerAlignV(unsigned AddrSpace) const override {
    return getPointerWidthV(AddrSpace);
  }

  uint64_t getMaxPointerWidth() const override {
    return getTriple().getArch() == llvm::Triple::amdgcn ? 64 : 32;
  }

  const char *getClobbers() const override { return ""; }

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return None;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      break;
    case 'v': // vgpr
    case 's': // sgpr
      Info.setAllowsRegister();
      return true;
    }
    return false;
  }

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeatureVec) const override;

  void adjustTargetOptions(const CodeGenOptions &CGOpts,
                           TargetOptions &TargetOpts) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override;

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::CharPtrBuiltinVaList;
  }

  static GPUKind parseR600Name(StringRef Name);

  static GPUKind parseAMDGCNName(StringRef Name);

  bool isValidCPUName(StringRef Name) const override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      return GK_NONE != parseAMDGCNName(Name);
    else
      return GK_NONE != parseR600Name(Name);
  }

  bool setCPU(const std::string &Name) override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      GPU = parseAMDGCNName(Name);
    else
      GPU = parseR600Name(Name);

    return GPU != GK_NONE;
  }

  void setSupportedOpenCLOpts() override {
    auto &Opts = getSupportedOpenCLOpts();
    Opts.support("cl_clang_storage_class_specifiers");
    Opts.support("cl_khr_icd");

    if (hasFP64)
      Opts.support("cl_khr_fp64");
    if (GPU >= GK_EVERGREEN) {
      Opts.support("cl_khr_byte_addressable_store");
      Opts.support("cl_khr_global_int32_base_atomics");
      Opts.support("cl_khr_global_int32_extended_atomics");
      Opts.support("cl_khr_local_int32_base_atomics");
      Opts.support("cl_khr_local_int32_extended_atomics");
    }
    if (GPU >= GK_GFX6) {
      Opts.support("cl_khr_fp16");
      Opts.support("cl_khr_int64_base_atomics");
      Opts.support("cl_khr_int64_extended_atomics");
      Opts.support("cl_khr_mipmap_image");
      Opts.support("cl_khr_subgroups");
      Opts.support("cl_khr_3d_image_writes");
      Opts.support("cl_amd_media_ops");
      Opts.support("cl_amd_media_ops2");
    }
  }

  LangAS::ID getOpenCLTypeAddrSpace(const Type *T) const override {
    auto BT = dyn_cast<BuiltinType>(T);

    if (!BT)
      return TargetInfo::getOpenCLTypeAddrSpace(T);

    switch (BT->getKind()) {
#define IMAGE_TYPE(ImgType, Id, SingletonId, Access, Suffix)                   \
  case BuiltinType::Id:                                                        \
    return LangAS::opencl_constant;
#include "clang/Basic/OpenCLImageTypes.def"
    case BuiltinType::OCLClkEvent:
    case BuiltinType::OCLQueue:
    case BuiltinType::OCLReserveID:
      return LangAS::opencl_global;

    default:
      return TargetInfo::getOpenCLTypeAddrSpace(T);
    }
  }

  llvm::Optional<unsigned> getConstantAddressSpace() const override {
    return LangAS::FirstTargetAddressSpace + AS.Constant;
  }

  /// \returns Target specific vtbl ptr address space.
  unsigned getVtblPtrAddressSpace() const override { return AS.Constant; }

  /// \returns If a target requires an address within a target specific address
  /// space \p AddressSpace to be converted in order to be used, then return the
  /// corresponding target specific DWARF address space.
  ///
  /// \returns Otherwise return None and no conversion will be emitted in the
  /// DWARF.
  Optional<unsigned>
  getDWARFAddressSpace(unsigned AddressSpace) const override {
    const unsigned DWARF_Private = 1;
    const unsigned DWARF_Local = 2;
    if (AddressSpace == AS.Private) {
      return DWARF_Private;
    } else if (AddressSpace == AS.Local) {
      return DWARF_Local;
    } else {
      return None;
    }
  }

  CallingConvCheckResult checkCallingConvention(CallingConv CC) const override {
    switch (CC) {
    default:
      return CCCR_Warning;
    case CC_C:
    case CC_OpenCLKernel:
      return CCCR_OK;
    }
  }

  // In amdgcn target the null pointer in global, constant, and generic
  // address space has value 0 but in private and local address space has
  // value ~0.
  uint64_t getNullPointerValue(unsigned AS) const override {
    return AS == LangAS::opencl_local ? ~0 : 0;
  }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_AMDGPU_H
