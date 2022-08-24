//===--- M88k.h - Declare M88k target feature support ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares M88k TargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H

#include "OSTargets.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY M88kTargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];

  enum CPUKind { CK_Unknown, CK_88000, CK_88100, CK_88110 } CPU = CK_Unknown;

public:
  M88kTargetInfo(const llvm::Triple &Triple, const TargetOptions &);

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;
  ArrayRef<Builtin::Info> getTargetBuiltins() const override;
  bool hasFeature(StringRef Feature) const override;
  ArrayRef<const char *> getGCCRegNames() const override;
  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;
  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override;
  const char *getClobbers() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::M88kBuiltinVaList;
  }

  bool setCPU(const std::string &Name) override;
  void fillValidCPUList(SmallVectorImpl<StringRef> &Values) const override;
  bool isValidCPUName(StringRef Name) const override;
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_M88K_H
