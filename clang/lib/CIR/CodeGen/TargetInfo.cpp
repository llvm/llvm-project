#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenFunction.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"

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
class AMDGPUABIInfo : public ABIInfo {
public:
  AMDGPUABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class AMDGPUTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AMDGPUTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<AMDGPUABIInfo>(cgt)) {}

  clang::LangAS
  getGlobalVarAddressSpace(CIRGenModule &cgm,
                           const clang::VarDecl *decl) const override {
    using clang::LangAS;
    assert(!cgm.getLangOpts().OpenCL &&
           !(cgm.getLangOpts().CUDA && cgm.getLangOpts().CUDAIsDevice) &&
           "Address space agnostic languages only");
    LangAS defaultGlobalAS = LangAS::opencl_global;
    if (!decl)
      return defaultGlobalAS;

    LangAS addrSpace = decl->getType().getAddressSpace();
    if (addrSpace != LangAS::Default)
      return addrSpace;

    // Only promote to address space 4 if VarDecl has constant initialization.
    if (decl->getType().isConstantStorage(cgm.getASTContext(), false, false) &&
        decl->hasConstantInitialization()) {
      if (auto constAS = cgm.getTarget().getConstantAddressSpace())
        return *constAS;
    }

    return defaultGlobalAS;
  }

  mlir::ptr::MemorySpaceAttrInterface
  getCIRAllocaAddressSpace() const override {
    return cir::LangAddressSpaceAttr::get(
        &getABIInfo().cgt.getMLIRContext(),
        cir::LangAddressSpace::OffloadPrivate);
  }
};
} // namespace

namespace {

class NVPTXABIInfo : public ABIInfo {
public:
  NVPTXABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}
};

class NVPTXTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  NVPTXTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<NVPTXABIInfo>(cgt)) {}
};
} // namespace

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createNVPTXTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<NVPTXTargetCIRGenInfo>(cgt);
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createAMDGPUTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<AMDGPUTargetCIRGenInfo>(cgt);
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

clang::LangAS
TargetCIRGenInfo::getGlobalVarAddressSpace(CIRGenModule &CGM,
                                           const clang::VarDecl *D) const {
  assert(!CGM.getLangOpts().OpenCL &&
         !(CGM.getLangOpts().CUDA && CGM.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  return D ? D->getType().getAddressSpace() : LangAS::Default;
}

mlir::Value TargetCIRGenInfo::performAddrSpaceCast(
    CIRGenFunction &cgf, mlir::Value v,
    mlir::ptr::MemorySpaceAttrInterface srcAS, mlir::Type destTy,
    bool isNonNull) const {
  // Since target may map different address spaces in AST to the same address
  // space, an address space conversion may end up as a bitcast.
  if (cir::GlobalOp globalOp = v.getDefiningOp<cir::GlobalOp>())
    cgf.cgm.errorNYI("Global op addrspace cast");
  // Try to preserve the source's name to make IR more readable.
  return cgf.getBuilder().createAddrSpaceCast(v, destTy);
}
