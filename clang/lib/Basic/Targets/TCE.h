//===--- TCE.h - Declare TCE target feature support -------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares TCE TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_TCE_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_TCE_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/Support/Compiler.h"
#include "llvm/TargetParser/Triple.h"

namespace clang {
namespace targets {

// llvm and clang cannot be used directly to output native binaries for
// target, but is used to compile C code to llvm bitcode with correct
// type and alignment information.
//
// TCE uses the llvm bitcode as input and uses it for generating customized
// target processor and program binary. TCE co-design environment is
// publicly available in http://tce.cs.tut.fi

static constexpr LangASMap TCEOpenCLAddrSpaceMap = {
    {LangAS::opencl_global, 1},
    {LangAS::opencl_local, 3},
    {LangAS::opencl_constant, 2},
    // FIXME: generic has to be added to the target
    {LangAS::opencl_generic, 0},
    {LangAS::opencl_global_device, 1},
    {LangAS::opencl_global_host, 1},
};

class LLVM_LIBRARY_VISIBILITY TCETargetInfo : public TargetInfo {
public:
  TCETargetInfo(const llvm::Triple &Triple, const TargetOptions &)
      : TargetInfo(Triple) {
    TLSSupported = false;
    IntWidth = 32;
    LongWidth = LongLongWidth = 32;
    PointerWidth = 32;
    IntAlign = 32;
    LongAlign = LongLongAlign = 32;
    PointerAlign = 32;
    SuitableAlign = 32;
    SizeType = UnsignedInt;
    IntMaxType = SignedLong;
    IntPtrType = SignedInt;
    PtrDiffType = SignedInt;
    FloatWidth = 32;
    FloatAlign = 32;
    DoubleWidth = 32;
    DoubleAlign = 32;
    LongDoubleWidth = 32;
    LongDoubleAlign = 32;
    FloatFormat = &llvm::APFloat::IEEEsingle();
    DoubleFormat = &llvm::APFloat::IEEEsingle();
    LongDoubleFormat = &llvm::APFloat::IEEEsingle();
    resetDataLayout("E-p:32:32:32-i1:8:8-i8:8:32-"
                    "i16:16:32-i32:32:32-i64:32:32-"
                    "f16:16:16-f32:32:32-f64:32:32-v64:64:64-"
                    "i128:128-"
                    "v128:128:128-v256:256:256-v512:512:512-"
                    "v1024:1024:1024-v2048:2048:2048-"
                    "v4096:4096:4096-a0:0:32-n32");
    AddrSpaceMap = &TCEOpenCLAddrSpaceMap;
    UseAddrSpaceMapMangling = true;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  bool hasFeature(StringRef Feature) const override { return Feature == "tce"; }

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override {
    return {};
  }

  std::string_view getClobbers() const override { return ""; }

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  ArrayRef<const char *> getGCCRegNames() const override { return {}; }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return true;
  }

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override {
    return {};
  }

  // TCE does not have fixed, but user specified register names.
  bool isValidGCCRegisterName(StringRef Name) const override { return true; }
};

class LLVM_LIBRARY_VISIBILITY TCELETargetInfo : public TCETargetInfo {
public:
  TCELETargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TCETargetInfo(Triple, Opts) {
    BigEndian = false;

    resetDataLayout("e-p:32:32:32-i1:8:8-i8:8:32-"
                    "i16:16:32-i32:32:32-i64:32:32-"
                    "f16:16:16-f32:32:32-f64:32:32-v64:64:64-"
                    "i128:128-"
                    "v128:128:128-v256:256:256-v512:512:512-"
                    "v1024:1024:1024-v2048:2048:2048-"
                    "v4096:4096:4096-a0:0:32-n32");
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

class LLVM_LIBRARY_VISIBILITY TCELE64TargetInfo : public TCETargetInfo {
public:
  TCELE64TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts)
      : TCETargetInfo(Triple, Opts) {
    BigEndian = false;

    resetDataLayout("e-p:64:64:64-i1:8:64-i8:8:64-"
                    "i16:16:64-i32:32:64-i64:64:64-"
                    "f16:16:64-f32:32:64-f64:64:64-v64:64:64-"
                    "i128:128-"
                    "v128:128:128-v256:256:256-v512:512:512-"
                    "v1024:1024:1024-v2048:2048:2048-"
                    "v4096:4096:4096-a0:0:64-n64");

    LongWidth = LongLongWidth = 64;
    PointerWidth = 64;
    PointerAlign = 64;
    LongAlign = LongLongAlign = 64;
    IntPtrType = SignedLong;
    SizeType = UnsignedLong;
    PtrDiffType = SignedLong;
    DoubleWidth = 64;
    DoubleAlign = 64;
    LongDoubleWidth = 64;
    LongDoubleAlign = 64;
    DoubleFormat = &llvm::APFloat::IEEEdouble();
    LongDoubleFormat = &llvm::APFloat::IEEEdouble();
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
};

} // namespace targets
} // namespace clang
#endif // LLVM_CLANG_LIB_BASIC_TARGETS_TCE_H
