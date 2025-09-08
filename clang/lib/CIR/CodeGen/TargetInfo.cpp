#include "TargetInfo.h"
#include "ABIInfo.h"

using namespace clang;
using namespace clang::CIRGen;

bool clang::CIRGen::isEmptyRecordForLayout(const ASTContext &context,
                                           QualType t) {
  const auto *rd = t->getAsRecordDecl();
  if (!rd)
    return false;

  // If this is a C++ record, check the bases first.
  if (const CXXRecordDecl *cxxrd = dyn_cast<CXXRecordDecl>(rd)) {
    if (cxxrd->isDynamicClass())
      return false;

    for (const auto &i : cxxrd->bases())
      if (!isEmptyRecordForLayout(context, i.getType()))
        return false;
  }

  for (const auto *i : rd->fields())
    if (!isEmptyFieldForLayout(context, i))
      return false;

  return true;
}

bool clang::CIRGen::isEmptyFieldForLayout(const ASTContext &context,
                                          const FieldDecl *fd) {
  if (fd->isZeroLengthBitField())
    return true;

  if (fd->isUnnamedBitField())
    return false;

  return isEmptyRecordForLayout(context, fd->getType());
}

namespace {

class X8664ABIInfo : public ABIInfo {
public:
  X8664ABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class X8664TargetCIRGenInfo : public TargetCIRGenInfo {
public:
  X8664TargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<X8664ABIInfo>(cgt)) {}
};

} // namespace

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
