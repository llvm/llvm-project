//===--- Parasol.h - Declare Parasol target feature support ---------*- C++ -*-===//
//
// Modified by Sunscreen under the AGPLv3 license; see the README at the
// repository root for more information
//
//===----------------------------------------------------------------------===//
//
// This file declares ParasolTargetInfo objects.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_Parasol_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_Parasol_H

#include "clang/Basic/TargetInfo.h"
#include "clang/Basic/TargetOptions.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY ParasolTargetInfo : public TargetInfo {
  static const char *const GCCRegNames[];

public:
  ParasolTargetInfo(const llvm::Triple &Triple, const TargetOptions &)
    : TargetInfo(Triple) {
    // Description string has to be kept in sync with backend string at
    // llvm/lib/Target/Parasol/ParasolTargetMachine.cpp
    resetDataLayout("e"
                    // ELF name mangling
                    "-m:e"
                    // 32-bit pointers, 32-bit aligned
                    "-p:32:32"
                    // 64-bit integers, 64-bit aligned
                    "-i64:64"
                    // 32-bit native integer width i.e register are 32-bit
                    "-n1:8:16:32"
                    // 128-bit natural stack alignment
                    "-S128"
    );
    SuitableAlign = 128;
    WCharType = SignedInt;
    WIntType = UnsignedInt;
    IntPtrType = SignedInt;
    PtrDiffType = SignedInt;
    SizeType = UnsignedInt;
  }

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<TargetInfo::GCCRegAlias> getGCCRegAliases() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override {
    return TargetInfo::VoidPtrBuiltinVaList;
  }

  ArrayRef<Builtin::Info> getTargetBuiltins() const override {
    return std::nullopt;
  }

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override {
    return false;
  }

  std::string_view getClobbers() const override {
    return "";
  }
};

} // namespace targets
} // namespace clang

#endif // LLVM_CLANG_LIB_BASIC_TARGETS_Parasol_H
