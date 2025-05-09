#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenFunctionInfo.h"
#include "clang/CIR/MissingFeatures.h"

using namespace clang;
using namespace clang::CIRGen;

static bool testIfIsVoidTy(QualType ty) {
  const auto *builtinTy = ty->getAs<BuiltinType>();
  return builtinTy && builtinTy->getKind() == BuiltinType::Void;
}

namespace {

class X8664ABIInfo : public ABIInfo {
public:
  X8664ABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  void computeInfo(CIRGenFunctionInfo &funcInfo) const override;
};

class X8664TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  X8664TargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<X8664ABIInfo>(cgt)) {}
};

} // namespace

void X8664ABIInfo::computeInfo(CIRGenFunctionInfo &funcInfo) const {
  // Top level CIR has unlimited arguments and return types. Lowering for ABI
  // specific concerns should happen during a lowering phase. Assume everything
  // is direct for now.
  for (CIRGenFunctionInfoArgInfo &info : funcInfo.arguments()) {
    if (testIfIsVoidTy(info.type))
      info.info = cir::ABIArgInfo::getIgnore();
    else
      info.info = cir::ABIArgInfo::getDirect(cgt.convertType(info.type));
  }

  CanQualType retTy = funcInfo.getReturnType();
  if (testIfIsVoidTy(retTy))
    funcInfo.getReturnInfo() = cir::ABIArgInfo::getIgnore();
  else
    funcInfo.getReturnInfo() =
        cir::ABIArgInfo::getDirect(cgt.convertType(retTy));
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createX8664TargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<X8664TargetCIRGenInfo>(cgt);
}

ABIInfo::~ABIInfo() noexcept = default;

bool TargetCIRGenInfo::isNoProtoCallVariadic(
    const FunctionNoProtoType *fnType) const {
  // The following conventions are known to require this to be false:
  //   x86_stdcall
  //   MIPS
  // For everything else, we just prefer false unless we opt out.
  return false;
}
