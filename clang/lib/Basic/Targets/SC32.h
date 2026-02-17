#ifndef LLVM_CLANG_LIB_BASIC_TARGETS_SC32_H
#define LLVM_CLANG_LIB_BASIC_TARGETS_SC32_H

#include "clang/Basic/TargetInfo.h"
#include "llvm/Support/Compiler.h"

namespace clang {
namespace targets {

class LLVM_LIBRARY_VISIBILITY SC32TargetInfo : public TargetInfo {
public:
  SC32TargetInfo(const llvm::Triple &Triple, const TargetOptions &Opts);

  void getTargetDefines(const LangOptions &Opts,
                        MacroBuilder &Builder) const override;

  llvm::SmallVector<Builtin::InfosShard> getTargetBuiltins() const override;

  BuiltinVaListKind getBuiltinVaListKind() const override;

  bool validateAsmConstraint(const char *&Name,
                             TargetInfo::ConstraintInfo &info) const override;

  std::string_view getClobbers() const override;

  ArrayRef<const char *> getGCCRegNames() const override;

  ArrayRef<GCCRegAlias> getGCCRegAliases() const override;
};

} // namespace targets
} // namespace clang

#endif
