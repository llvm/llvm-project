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

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/StringSet.h"
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
  };

  struct GPUInfo {
    llvm::StringLiteral Name;
    llvm::StringLiteral CanonicalName;
    AMDGPUTargetInfo::GPUKind Kind;
  };

  GPUInfo GPU;

  static constexpr GPUInfo InvalidGPU = {{""}, {""}, GK_NONE};
  static constexpr GPUInfo R600Names[26] = {
      {{"r600"},    {"r600"},    GK_R600},
      {{"rv630"},   {"r600"},    GK_R600},
      {{"rv635"},   {"r600"},    GK_R600},
      {{"r630"},    {"r630"},    GK_R600},
      {{"rs780"},   {"rs880"},   GK_R600},
      {{"rs880"},   {"rs880"},   GK_R600},
      {{"rv610"},   {"rs880"},   GK_R600},
      {{"rv620"},   {"rs880"},   GK_R600},
      {{"rv670"},   {"rv670"},   GK_R600_DOUBLE_OPS},
      {{"rv710"},   {"rv710"},   GK_R700},
      {{"rv730"},   {"rv730"},   GK_R700},
      {{"rv740"},   {"rv770"},   GK_R700_DOUBLE_OPS},
      {{"rv770"},   {"rv770"},   GK_R700_DOUBLE_OPS},
      {{"cedar"},   {"cedar"},   GK_EVERGREEN},
      {{"palm"},    {"cedar"},   GK_EVERGREEN},
      {{"cypress"}, {"cypress"}, GK_EVERGREEN_DOUBLE_OPS},
      {{"hemlock"}, {"cypress"}, GK_EVERGREEN_DOUBLE_OPS},
      {{"juniper"}, {"juniper"}, GK_EVERGREEN},
      {{"redwood"}, {"redwood"}, GK_EVERGREEN},
      {{"sumo"},    {"sumo"},    GK_EVERGREEN},
      {{"sumo2"},   {"sumo"},    GK_EVERGREEN},
      {{"barts"},   {"barts"},   GK_NORTHERN_ISLANDS},
      {{"caicos"},  {"caicos"},  GK_NORTHERN_ISLANDS},
      {{"turks"},   {"turks"},   GK_NORTHERN_ISLANDS},
      {{"aruba"},   {"cayman"},  GK_CAYMAN},
      {{"cayman"},  {"cayman"},  GK_CAYMAN},
  };
  static constexpr GPUInfo AMDGCNNames[30] = {
      {{"gfx600"},    {"gfx600"}, GK_GFX6},
      {{"tahiti"},    {"gfx600"}, GK_GFX6},
      {{"gfx601"},    {"gfx601"}, GK_GFX6},
      {{"hainan"},    {"gfx601"}, GK_GFX6},
      {{"oland"},     {"gfx601"}, GK_GFX6},
      {{"pitcairn"},  {"gfx601"}, GK_GFX6},
      {{"verde"},     {"gfx601"}, GK_GFX6},
      {{"gfx700"},    {"gfx700"}, GK_GFX7},
      {{"kaveri"},    {"gfx700"}, GK_GFX7},
      {{"gfx701"},    {"gfx701"}, GK_GFX7},
      {{"hawaii"},    {"gfx701"}, GK_GFX7},
      {{"gfx702"},    {"gfx702"}, GK_GFX7},
      {{"gfx703"},    {"gfx703"}, GK_GFX7},
      {{"kabini"},    {"gfx703"}, GK_GFX7},
      {{"mullins"},   {"gfx703"}, GK_GFX7},
      {{"gfx704"},    {"gfx704"}, GK_GFX7},
      {{"bonaire"},   {"gfx704"}, GK_GFX7},
      {{"gfx801"},    {"gfx801"}, GK_GFX8},
      {{"carrizo"},   {"gfx801"}, GK_GFX8},
      {{"gfx802"},    {"gfx802"}, GK_GFX8},
      {{"iceland"},   {"gfx802"}, GK_GFX8},
      {{"tonga"},     {"gfx802"}, GK_GFX8},
      {{"gfx803"},    {"gfx803"}, GK_GFX8},
      {{"fiji"},      {"gfx803"}, GK_GFX8},
      {{"polaris10"}, {"gfx803"}, GK_GFX8},
      {{"polaris11"}, {"gfx803"}, GK_GFX8},
      {{"gfx810"},    {"gfx810"}, GK_GFX8},
      {{"stoney"},    {"gfx810"}, GK_GFX8},
      {{"gfx900"},    {"gfx900"}, GK_GFX9},
      {{"gfx902"},    {"gfx902"}, GK_GFX9},
  };

  bool hasFP64 : 1;
  bool hasFMAF : 1;
  bool hasLDEXPF : 1;
  const AddrSpace AS;

  static bool hasFullSpeedFMAF32(StringRef GPUName) {
    return parseAMDGCNName(GPUName).Kind >= GK_GFX9;
  }

  static bool isAMDGCN(const llvm::Triple &TT) {
    return TT.getArch() == llvm::Triple::amdgcn;
  }

  static bool isGenericZero(const llvm::Triple &TT) { return true; }

public:
  AMDGPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void setAddressSpaceMap(bool DefaultIsPrivate);

  void adjust(LangOptions &Opts) override;

  uint64_t getPointerWidthV(unsigned AddrSpace) const override {
    if (GPU.Kind <= GK_CAYMAN)
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

  /// Accepted register names: (n, m is unsigned integer, n < m)
  /// v
  /// s
  /// {vn}, {v[n]}
  /// {sn}, {s[n]}
  /// {S} , where S is a special register name
  ////{v[n:m]}
  /// {s[n:m]}
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    static const ::llvm::StringSet<> SpecialRegs({
        "exec", "vcc", "flat_scratch", "m0", "scc", "tba", "tma",
        "flat_scratch_lo", "flat_scratch_hi", "vcc_lo", "vcc_hi", "exec_lo",
        "exec_hi", "tma_lo", "tma_hi", "tba_lo", "tba_hi",
    });

    StringRef S(Name);
    bool HasLeftParen = false;
    if (S.front() == '{') {
      HasLeftParen = true;
      S = S.drop_front();
    }
    if (S.empty())
      return false;
    if (S.front() != 'v' && S.front() != 's') {
      if (!HasLeftParen)
        return false;
      auto E = S.find('}');
      if (!SpecialRegs.count(S.substr(0, E)))
        return false;
      S = S.drop_front(E + 1);
      if (!S.empty())
        return false;
      // Found {S} where S is a special register.
      Info.setAllowsRegister();
      Name = S.data() - 1;
      return true;
    }
    S = S.drop_front();
    if (!HasLeftParen) {
      if (!S.empty())
        return false;
      // Found s or v.
      Info.setAllowsRegister();
      Name = S.data() - 1;
      return true;
    }
    bool HasLeftBracket = false;
    if (!S.empty() && S.front() == '[') {
      HasLeftBracket = true;
      S = S.drop_front();
    }
    unsigned long long N;
    if (S.empty() || consumeUnsignedInteger(S, 10, N))
      return false;
    if (!S.empty() && S.front() == ':') {
      if (!HasLeftBracket)
        return false;
      S = S.drop_front();
      unsigned long long M;
      if (consumeUnsignedInteger(S, 10, M) || N >= M)
        return false;
    }
    if (HasLeftBracket) {
      if (S.empty() || S.front() != ']')
        return false;
      S = S.drop_front();
    }
    if (S.empty() || S.front() != '}')
      return false;
    S = S.drop_front();
    if (!S.empty())
      return false;
    // Found {vn}, {sn}, {v[n]}, {s[n]}, {v[n:m]}, or {s[n:m]}.
    Info.setAllowsRegister();
    Name = S.data() - 1;
    return true;
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

  static GPUInfo parseR600Name(StringRef Name);

  static GPUInfo parseAMDGCNName(StringRef Name);

  bool isValidCPUName(StringRef Name) const override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      return GK_NONE != parseAMDGCNName(Name).Kind;
    else
      return GK_NONE != parseR600Name(Name).Kind;
  }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;

  bool setCPU(const std::string &Name) override {
    if (getTriple().getArch() == llvm::Triple::amdgcn)
      GPU = parseAMDGCNName(Name);
    else
      GPU = parseR600Name(Name);

    return GPU.Kind != GK_NONE;
  }

  void setSupportedOpenCLOpts() override {
    auto &Opts = getSupportedOpenCLOpts();
    Opts.support("cl_clang_storage_class_specifiers");
    Opts.support("cl_khr_icd");

    if (hasFP64)
      Opts.support("cl_khr_fp64");
    if (GPU.Kind >= GK_EVERGREEN) {
      Opts.support("cl_khr_byte_addressable_store");
      Opts.support("cl_khr_global_int32_base_atomics");
      Opts.support("cl_khr_global_int32_extended_atomics");
      Opts.support("cl_khr_local_int32_base_atomics");
      Opts.support("cl_khr_local_int32_extended_atomics");
    }
    if (GPU.Kind >= GK_GFX6) {
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

  LangAS getOpenCLTypeAddrSpace(OpenCLTypeKind TK) const override {
    switch (TK) {
    case OCLTK_Image:
      return LangAS::opencl_constant;

    case OCLTK_ClkEvent:
    case OCLTK_Queue:
    case OCLTK_ReserveID:
      return LangAS::opencl_global;

    default:
      return TargetInfo::getOpenCLTypeAddrSpace(TK);
    }
  }

  llvm::Optional<LangAS> getConstantAddressSpace() const override {
    return getLangASFromTargetAS(AS.Constant);
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
  uint64_t getNullPointerValue(LangAS AS) const override {
    return AS == LangAS::opencl_local ? ~0 : 0;
  }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_AMDGPU_H
