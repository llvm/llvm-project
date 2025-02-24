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

  mlir::LLVM::TBAATypeDescriptorAttr
  lowerCIRTBAAAttrToLLVMTBAAAttr(mlir::Attribute tbaa) {
    if (auto charAttr = mlir::dyn_cast<cir::TBAAOmnipotentCharAttr>(tbaa)) {
      return getChar();
    }
    if (auto scalarAttr = mlir::dyn_cast<cir::TBAAScalarAttr>(tbaa)) {
      mlir::DataLayout layout;
      auto size = layout.getTypeSize(scalarAttr.getType());
      return createScalarTypeNode(scalarAttr.getId(), getChar(), size);
    }
    if (auto structAttr = mlir::dyn_cast<cir::TBAAStructAttr>(tbaa)) {
      llvm::SmallVector<mlir::LLVM::TBAAMemberAttr, 4> members;
      for (const auto &member : structAttr.getMembers()) {
        auto memberTypeDesc =
            lowerCIRTBAAAttrToLLVMTBAAAttr(member.getTypeDesc());
        auto memberAttr = mlir::LLVM::TBAAMemberAttr::get(
            mlirContext, memberTypeDesc, member.getOffset());
        members.push_back(memberAttr);
      }
      return mlir::LLVM::TBAATypeDescriptorAttr::get(
          mlirContext, structAttr.getId(), members);
    }
    return nullptr;
  }

private:
  mlir::LLVM::TBAARootAttr getRoot() {
    return createTBAARoot("Simple C/C++ TBAA");
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

  mlir::LLVM::TBAARootAttr createTBAARoot(llvm::StringRef name) {
    return mlir::LLVM::TBAARootAttr::get(
        mlirContext, mlir::StringAttr::get(mlirContext, name));
  }

  mlir::MLIRContext *mlirContext;
};

mlir::ArrayAttr lowerCIRTBAAAttr(mlir::Attribute tbaa,
                                 mlir::ConversionPatternRewriter &rewriter,
                                 cir::LowerModule *lowerMod) {
  auto *ctx = rewriter.getContext();
  CIRToLLVMTBAAAttrLowering lower(ctx);
  if (auto tbaaTag = mlir::dyn_cast<cir::TBAATagAttr>(tbaa)) {
    mlir::LLVM::TBAATypeDescriptorAttr accessType =
        lower.lowerCIRTBAAAttrToLLVMTBAAAttr(tbaaTag.getAccess());
    if (auto structAttr =
            mlir::dyn_cast<cir::TBAAStructAttr>(tbaaTag.getBase())) {
      auto baseType = lower.lowerCIRTBAAAttrToLLVMTBAAAttr(structAttr);
      auto tag = mlir::LLVM::TBAATagAttr::get(baseType, accessType,
                                              tbaaTag.getOffset());
      return mlir::ArrayAttr::get(ctx, {tag});
    }
  } else {
    auto accessType = lower.lowerCIRTBAAAttrToLLVMTBAAAttr(tbaa);
    if (accessType) {
      auto tag = mlir::LLVM::TBAATagAttr::get(accessType, accessType, 0);
      return mlir::ArrayAttr::get(ctx, {tag});
    }
  }
  return mlir::ArrayAttr();
}

} // namespace direct
} // namespace cir