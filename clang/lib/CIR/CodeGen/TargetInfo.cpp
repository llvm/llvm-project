#include "TargetInfo.h"
#include "ABIInfo.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "mlir/Dialect/Ptr/IR/MemorySpaceInterfaces.h"
#include "clang/Basic/AddressSpaces.h"
#include "clang/Basic/LangOptions.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/MissingFeatures.h"

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

class AMDGPUABIInfo : public ABIInfo {
public:
  AMDGPUABIInfo(CIRGenTypes &cgt) : ABIInfo(cgt) {}

  cir::VectorType
  getOptimalVectorMemoryType(cir::VectorType type,
                             const clang::LangOptions &langOpts) const override;
};

class AMDGPUTargetCIRGenInfo : public TargetCIRGenInfo {
public:
  AMDGPUTargetCIRGenInfo(CIRGenTypes &cgt)
      : TargetCIRGenInfo(std::make_unique<AMDGPUABIInfo>(cgt)) {}

  bool supportsLibCall() const override { return false; }

  void setTargetAttributes(const clang::Decl *decl, mlir::Operation *global,
                           CIRGenModule &cgm) const override {
    if (auto func = mlir::dyn_cast<cir::FuncOp>(global)) {
      if (requiresAMDGPUProtectedVisibility(decl, func.getGlobalVisibility())) {
        func.setGlobalVisibility(cir::VisibilityKind::Protected);
        func.setDSOLocal(true);
      }
      setAMDGPUTargetFunctionAttributes(decl, func, cgm);
    } else if (auto gv = mlir::dyn_cast<cir::GlobalOp>(global)) {
      if (requiresAMDGPUProtectedVisibility(decl, gv.getGlobalVisibility())) {
        gv.setGlobalVisibility(cir::VisibilityKind::Protected);
        gv.setDSOLocal(true);
      }
    }
  }

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
        decl->hasConstantInitialization())
      return LangAS::opencl_constant;

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
clang::CIRGen::createAMDGPUTargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<AMDGPUTargetCIRGenInfo>(cgt);
}

std::unique_ptr<TargetCIRGenInfo>
clang::CIRGen::createX8664TargetCIRGenInfo(CIRGenTypes &cgt) {
  return std::make_unique<X8664TargetCIRGenInfo>(cgt);
}

cir::VectorType
ABIInfo::getOptimalVectorMemoryType(cir::VectorType type,
                                    const clang::LangOptions &langOpts) const {
  if (type.getSize() == 3 && !langOpts.PreserveVec3Type)
    return cir::VectorType::get(type.getElementType(), 4);
  return type;
}

cir::VectorType AMDGPUABIInfo::getOptimalVectorMemoryType(
    cir::VectorType type, const clang::LangOptions &langOpts) const {
  // AMDGPU has legal instructions for 96-bit vectors, so 3 x 32-bit vectors
  // do not need to be widened for memory operations.
  // FIXME: This check should be a subtarget feature as technically SI doesn't
  // support it.
  if (type.getSize() == 3 &&
      cgt.getCGModule().getDataLayout().getTypeSizeInBits(type) == 96)
    return type;
  return ABIInfo::getOptimalVectorMemoryType(type, langOpts);
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
TargetCIRGenInfo::getGlobalVarAddressSpace(CIRGenModule &cgm,
                                           const clang::VarDecl *d) const {
  assert(!cgm.getLangOpts().OpenCL &&
         !(cgm.getLangOpts().CUDA && cgm.getLangOpts().CUDAIsDevice) &&
         "Address space agnostic languages only");
  return d ? d->getType().getAddressSpace() : LangAS::Default;
}
