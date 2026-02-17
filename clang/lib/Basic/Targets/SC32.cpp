#include "SC32.h"

using namespace clang;
using namespace clang::targets;

SC32TargetInfo::SC32TargetInfo(const llvm::Triple &Triple,
                               const TargetOptions &Opts)
    : TargetInfo(Triple) {
  resetDataLayout();
}

void SC32TargetInfo::getTargetDefines(const LangOptions &Opts,
                                      MacroBuilder &Builder) const {}

llvm::SmallVector<Builtin::InfosShard>
SC32TargetInfo::getTargetBuiltins() const {
  return {};
}

TargetInfo::BuiltinVaListKind SC32TargetInfo::getBuiltinVaListKind() const {
  return TargetInfo::VoidPtrBuiltinVaList;
}

bool SC32TargetInfo::validateAsmConstraint(
    const char *&Name, TargetInfo::ConstraintInfo &info) const {
  return false;
}

std::string_view SC32TargetInfo::getClobbers() const { return ""; }

ArrayRef<const char *> SC32TargetInfo::getGCCRegNames() const { return {}; }

ArrayRef<TargetInfo::GCCRegAlias> SC32TargetInfo::getGCCRegAliases() const {
  return {};
}
