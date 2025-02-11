#include "LowerToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

namespace cir {
namespace direct {

class CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAAttrLowering(mlir::MLIRContext *mlirContext)
      : mlirContext(mlirContext) {}
  mlir::LLVM::TBAARootAttr getRoot() {
    return mlir::LLVM::TBAARootAttr::get(
        mlirContext, mlir::StringAttr::get(mlirContext, "Simple C/C++ TBAA"));
  }

  mlir::LLVM::TBAATypeDescriptorAttr getChar() {
    return createScalarTypeNode("omnipotent char", getRoot(), 0);
  }

  mlir::LLVM::TBAATypeDescriptorAttr
  createScalarTypeNode(llvm::StringRef typeName,
                       mlir::LLVM::TBAANodeAttr parent, int64_t size) {
    llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 2> members;
    members.push_back(mlir::LLVM::TBAAMemberAttr::get(mlirContext, parent, 0));
    return mlir::LLVM::TBAATypeDescriptorAttr::get(
        mlirContext, typeName,
        llvm::ArrayRef<mlir::LLVM::TBAAMemberAttr>(members));
  }

protected:
  mlir::MLIRContext *mlirContext;
};

class CIRToLLVMTBAAScalarAttrLowering : public CIRToLLVMTBAAAttrLowering {
public:
  CIRToLLVMTBAAScalarAttrLowering(mlir::MLIRContext *mlirContext)
      : CIRToLLVMTBAAAttrLowering(mlirContext) {}
  mlir::LLVM::TBAATypeDescriptorAttr
  lowerScalarType(cir::TBAAScalarAttr scalarAttr) {
    mlir::DataLayout layout;
    auto size = layout.getTypeSize(scalarAttr.getType());
    return createScalarTypeNode(scalarAttr.getId(), getChar(), size);
  }
};

mlir::ArrayAttr lowerCIRTBAAAttr(mlir::Attribute tbaa,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 cir::LowerModule *lowerMod) {
  auto *ctx = rewriter.getContext();
  CIRToLLVMTBAAScalarAttrLowering scalarLower(ctx);
  if (auto charAttr = mlir::dyn_cast<cir::TBAAOmnipotentCharAttr>(tbaa)) {
    auto accessType = scalarLower.getChar();
    auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
    return mlir::ArrayAttr::get(ctx, {tag});
  }
  if (auto scalarAttr = mlir::dyn_cast<cir::TBAAScalarAttr>(tbaa)) {
    auto accessType = scalarLower.lowerScalarType(scalarAttr);
    auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
    return mlir::ArrayAttr::get(ctx, {tag});
  }
  return mlir::ArrayAttr();
}

} // namespace direct
} // namespace cir