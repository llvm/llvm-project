//===--- Nios2.h - Declare Nios2 target feature support ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file declares Nios2 TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_NIOS2_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_NIOS2_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY Nios2TargetInfo : public TargetInfo {
  void setDataLayout() {
    if (BigEndian)
      resetDataLayout("E-p:32:32:32-i8:8:32-i16:16:32-n32");
    else
      resetDataLayout("e-p:32:32:32-i8:8:32-i16:16:32-n32");
  }

  static const Builtin::Info BuiltinInfo[];
  std::string CPU;
  std::string ABI;

public:
  Nios2TargetInfo(const llvm::Triple &triple, const TargetOptions &opts)
      : TargetInfo(triple), CPU(opts.CPU), ABI(opts.ABI) {
    SizeType = UnsignedInt;
    PtrDiffType = SignedInt;
    MaxAtomicPromoteWidth = MaxAtomicInlineWidth = 32;
    setDataLayout();
  }

  StringRef getABI() const override { return ABI; }
  bool setABI(const std::string &Name) override {
    if (Name == "o32" || Name == "eabi") {
      ABI = Name;
      return true;
    }
    return false;
  }

  bool isValidCPUName(StringRef Name) const override {
    return Name == "nios2r1" || Name == "nios2r2";
  }

  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override {
    Values.append({"nios2r1", "nios2r2"});
  }

  bool setCPU(const std::string &Name) override {
    if (isValidCPUName(Name)) {
      CPU = Name;
      return true;
    }
    return false;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<Builtin::Info> getTargetBuiltins() const override;

  bool isFeatureSupportedByCPU(StringRef Feature, StringRef CPU) const;

  bool
  initFeatureMap(llvm::StringMap<bool> &Features, DiagnosticsEngine &Diags,
                 StringRef CPU,
                 const std::vector<std::string> &FeatureVec) const override {
    static const char *allFeatures[] = {"nios2r2mandatory", "nios2r2bmx",
                                        "nios2r2mpx", "nios2r2cdx"
    };
    for (const char *feature : allFeatures) {
      Features[feature] = isFeatureSupportedByCPU(feature, CPU);
    }
    return true;
  }

  bool hasFeature(StringRef Feature) const override {
    return isFeatureSupportedByCPU(Feature, CPU);
  }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  ArrayRef<const char *> getGCCRegNames() const override {
    static const char *const GCCRegNames[] = {
        // CPU register names
        // Must match second column of GCCRegAliases
        "r0", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10",
        "r11", "r12", "r13", "r14", "r15", "r16", "r17", "r18", "r19", "r20",
        "r21", "r22", "r23", "r24", "r25", "r26", "r27", "r28", "r29", "r30",
        "r31",
        // Floating point register names
        "ctl0", "ctl1", "ctl2", "ctl3", "ctl4", "ctl5", "ctl6", "ctl7", "ctl8",
        "ctl9", "ctl10", "ctl11", "ctl12", "ctl13", "ctl14", "ctl15"
    };
    return llvm::makeArrayRef(GCCRegNames);
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &Info) const override {
    switch (*Name) {
    default:
      return false;

    case 'r': // CPU registers.
    case 'd': // Equivalent to "r" unless generating MIPS16 code.
    case 'y': // Equivalent to "r", backwards compatibility only.
    case 'f': // floating-point registers.
    case 'c': // $25 for indirect jumps
    case 'l': // lo register
    case 'x': // hilo register pair
      Info.setAllowsRegister();
      return true;
    }
  }

  const char *getClobbers() const override { return ""; }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    static const TargetInfo::GCCRegAlias aliases[] = {
        {{"zero"}, "r0"},      {{"at"}, "r1"},          {{"et"}, "r24"},
        {{"bt"}, "r25"},       {{"gp"}, "r26"},         {{"sp"}, "r27"},
        {{"fp"}, "r28"},       {{"ea"}, "r29"},         {{"ba"}, "r30"},
        {{"ra"}, "r31"},       {{"status"}, "ctl0"},    {{"estatus"}, "ctl1"},
        {{"bstatus"}, "ctl2"}, {{"ienable"}, "ctl3"},   {{"ipending"}, "ctl4"},
        {{"cpuid"}, "ctl5"},   {{"exception"}, "ctl7"}, {{"pteaddr"}, "ctl8"},
        {{"tlbacc"}, "ctl9"},  {{"tlbmisc"}, "ctl10"},  {{"badaddr"}, "ctl12"},
        {{"config"}, "ctl13"}, {{"mpubase"}, "ctl14"},  {{"mpuacc"}, "ctl15"},
    };
    return llvm::makeArrayRef(aliases);
  }
};

} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_NIOS2_H
