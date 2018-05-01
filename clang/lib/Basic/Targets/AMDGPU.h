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

  enum AddrSpace {
    Generic = 0,
    Global = 1,
    Local = 3,
    Constant = 4,
    Private = 5
  };
  static const LangASMap AMDGPUDefIsGenMap;
  static const LangASMap AMDGPUDefIsPrivMap;

  /// \brief GPU kinds supported by the AMDGPU target.
  enum GPUKind : uint32_t {
    // Not specified processor.
    GK_NONE = 0,

    // R600-based processors.
    GK_R600,
    GK_R630,
    GK_RS880,
    GK_RV670,
    GK_RV710,
    GK_RV730,
    GK_RV770,
    GK_CEDAR,
    GK_CYPRESS,
    GK_JUNIPER,
    GK_REDWOOD,
    GK_SUMO,
    GK_BARTS,
    GK_CAICOS,
    GK_CAYMAN,
    GK_TURKS,

    GK_R600_FIRST = GK_R600,
    GK_R600_LAST = GK_TURKS,

    // AMDGCN-based processors.
    GK_GFX600,
    GK_GFX601,
    GK_GFX700,
    GK_GFX701,
    GK_GFX702,
    GK_GFX703,
    GK_GFX704,
    GK_GFX801,
    GK_GFX802,
    GK_GFX803,
    GK_GFX810,
    GK_GFX900,
    GK_GFX902,
    GK_GFX904,
    GK_GFX906,

    GK_AMDGCN_FIRST = GK_GFX600,
    GK_AMDGCN_LAST = GK_GFX906,
  };

  struct GPUInfo {
    llvm::StringLiteral Name;
    llvm::StringLiteral CanonicalName;
    AMDGPUTargetInfo::GPUKind Kind;
    bool HasFMAF;
    bool HasFastFMAF;
    bool HasLDEXPF;
    bool HasFP64;
    bool HasFastFMA;
  };

  static constexpr GPUInfo InvalidGPU =
    {{""}, {""}, GK_NONE, false, false, false, false, false};
  static constexpr GPUInfo R600GPUs[26] = {
  // Name         Canonical    Kind        Has    Has    Has    Has    Has
  //              Name                     FMAF   Fast   LDEXPF FP64   Fast
  //                                              FMAF                 FMA
    {{"r600"},    {"r600"},    GK_R600,    false, false, false, false, false},
    {{"rv630"},   {"r600"},    GK_R600,    false, false, false, false, false},
    {{"rv635"},   {"r600"},    GK_R600,    false, false, false, false, false},
    {{"r630"},    {"r630"},    GK_R630,    false, false, false, false, false},
    {{"rs780"},   {"rs880"},   GK_RS880,   false, false, false, false, false},
    {{"rs880"},   {"rs880"},   GK_RS880,   false, false, false, false, false},
    {{"rv610"},   {"rs880"},   GK_RS880,   false, false, false, false, false},
    {{"rv620"},   {"rs880"},   GK_RS880,   false, false, false, false, false},
    {{"rv670"},   {"rv670"},   GK_RV670,   false, false, false, false, false},
    {{"rv710"},   {"rv710"},   GK_RV710,   false, false, false, false, false},
    {{"rv730"},   {"rv730"},   GK_RV730,   false, false, false, false, false},
    {{"rv740"},   {"rv770"},   GK_RV770,   false, false, false, false, false},
    {{"rv770"},   {"rv770"},   GK_RV770,   false, false, false, false, false},
    {{"cedar"},   {"cedar"},   GK_CEDAR,   false, false, false, false, false},
    {{"palm"},    {"cedar"},   GK_CEDAR,   false, false, false, false, false},
    {{"cypress"}, {"cypress"}, GK_CYPRESS, true,  false, false, false, false},
    {{"hemlock"}, {"cypress"}, GK_CYPRESS, true,  false, false, false, false},
    {{"juniper"}, {"juniper"}, GK_JUNIPER, false, false, false, false, false},
    {{"redwood"}, {"redwood"}, GK_REDWOOD, false, false, false, false, false},
    {{"sumo"},    {"sumo"},    GK_SUMO,    false, false, false, false, false},
    {{"sumo2"},   {"sumo"},    GK_SUMO,    false, false, false, false, false},
    {{"barts"},   {"barts"},   GK_BARTS,   false, false, false, false, false},
    {{"caicos"},  {"caicos"},  GK_BARTS,   false, false, false, false, false},
    {{"aruba"},   {"cayman"},  GK_CAYMAN,  true,  false, false, false, false},
    {{"cayman"},  {"cayman"},  GK_CAYMAN,  true,  false, false, false, false},
    {{"turks"},   {"turks"},   GK_TURKS,   false, false, false, false, false},
  };
  static constexpr GPUInfo AMDGCNGPUs[32] = {
  // Name           Canonical    Kind        Has   Has    Has    Has   Has
  //                Name                     FMAF  Fast   LDEXPF FP64  Fast
  //                                               FMAF                FMA
    {{"gfx600"},    {"gfx600"},  GK_GFX600,  true, true,  true,  true, true},
    {{"tahiti"},    {"gfx600"},  GK_GFX600,  true, true,  true,  true, true},
    {{"gfx601"},    {"gfx601"},  GK_GFX601,  true, false, true,  true, true},
    {{"hainan"},    {"gfx601"},  GK_GFX601,  true, false, true,  true, true},
    {{"oland"},     {"gfx601"},  GK_GFX601,  true, false, true,  true, true},
    {{"pitcairn"},  {"gfx601"},  GK_GFX601,  true, false, true,  true, true},
    {{"verde"},     {"gfx601"},  GK_GFX601,  true, false, true,  true, true},
    {{"gfx700"},    {"gfx700"},  GK_GFX700,  true, false, true,  true, true},
    {{"kaveri"},    {"gfx700"},  GK_GFX700,  true, false, true,  true, true},
    {{"gfx701"},    {"gfx701"},  GK_GFX701,  true, true,  true,  true, true},
    {{"hawaii"},    {"gfx701"},  GK_GFX701,  true, true,  true,  true, true},
    {{"gfx702"},    {"gfx702"},  GK_GFX702,  true, true,  true,  true, true},
    {{"gfx703"},    {"gfx703"},  GK_GFX703,  true, false, true,  true, true},
    {{"kabini"},    {"gfx703"},  GK_GFX703,  true, false, true,  true, true},
    {{"mullins"},   {"gfx703"},  GK_GFX703,  true, false, true,  true, true},
    {{"gfx704"},    {"gfx704"},  GK_GFX704,  true, false, true,  true, true},
    {{"bonaire"},   {"gfx704"},  GK_GFX704,  true, false, true,  true, true},
    {{"gfx801"},    {"gfx801"},  GK_GFX801,  true, true,  true,  true, true},
    {{"carrizo"},   {"gfx801"},  GK_GFX801,  true, true,  true,  true, true},
    {{"gfx802"},    {"gfx802"},  GK_GFX802,  true, false, true,  true, true},
    {{"iceland"},   {"gfx802"},  GK_GFX802,  true, false, true,  true, true},
    {{"tonga"},     {"gfx802"},  GK_GFX802,  true, false, true,  true, true},
    {{"gfx803"},    {"gfx803"},  GK_GFX803,  true, false, true,  true, true},
    {{"fiji"},      {"gfx803"},  GK_GFX803,  true, false, true,  true, true},
    {{"polaris10"}, {"gfx803"},  GK_GFX803,  true, false, true,  true, true},
    {{"polaris11"}, {"gfx803"},  GK_GFX803,  true, false, true,  true, true},
    {{"gfx810"},    {"gfx810"},  GK_GFX810,  true, false, true,  true, true},
    {{"stoney"},    {"gfx810"},  GK_GFX810,  true, false, true,  true, true},
    {{"gfx900"},    {"gfx900"},  GK_GFX900,  true, true,  true,  true, true},
    {{"gfx902"},    {"gfx902"},  GK_GFX900,  true, true,  true,  true, true},
    {{"gfx904"},    {"gfx904"},  GK_GFX904,  true, true,  true,  true, true},
    {{"gfx906"},    {"gfx906"},  GK_GFX906,  true, true,  true,  true, true},
  };

  static GPUInfo parseR600Name(StringRef Name);

  static GPUInfo parseAMDGCNName(StringRef Name);

  GPUInfo parseGPUName(StringRef Name) const;

  GPUInfo GPU;

  static bool isAMDGCN(const llvm::Triple &TT) {
    return TT.getArch() == llvm::Triple::amdgcn;
  }

public:
  AMDGPUTargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void setAddressSpaceMap(bool DefaultIsPrivate);

  void adjust(LangOptions &Opts) override;

  uint64_t getPointerWidthV(unsigned AddrSpace) const override {
    if (GPU.Kind <= GK_R600_LAST)
      return 32;
    if (AddrSpace == Private || AddrSpace == Local)
      return 32;
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

  // \p Constraint will be left pointing at the last character of
  // the constraint.  In practice, it won't be changed unless the
  // constraint is longer than one character.
  std::string convertConstraint(const char *&Constraint) const override {
    const char *Begin = Constraint;
    TargetInfo::ConstraintInfo Info("", "");
    if (validateAsmConstraint(Constraint, Info))
      return std::string(Begin).substr(0, Constraint - Begin + 1);

    Constraint = Begin;
    return std::string(1, *Constraint);
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

    return GK_NONE != GPU.Kind;
  }

  void setSupportedOpenCLOpts() override {
    auto &Opts = getSupportedOpenCLOpts();
    Opts.support("cl_clang_storage_class_specifiers");
    Opts.support("cl_khr_icd");

    if (GPU.HasFP64)
      Opts.support("cl_khr_fp64");
    if (GPU.Kind >= GK_CEDAR) {
      Opts.support("cl_khr_byte_addressable_store");
      Opts.support("cl_khr_global_int32_base_atomics");
      Opts.support("cl_khr_global_int32_extended_atomics");
      Opts.support("cl_khr_local_int32_base_atomics");
      Opts.support("cl_khr_local_int32_extended_atomics");
    }
    if (GPU.Kind >= GK_AMDGCN_FIRST) {
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
    return getLangASFromTargetAS(Constant);
  }

  /// \returns Target specific vtbl ptr address space.
  unsigned getVtblPtrAddressSpace() const override {
    return static_cast<unsigned>(Constant);
  }

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
    if (AddressSpace == Private) {
      return DWARF_Private;
    } else if (AddressSpace == Local) {
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
