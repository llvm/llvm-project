//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenMP extensions as applied to CIR dialect.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/OpenMP/RegisterOpenMPExtensions.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPInterfaces.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"

namespace {
struct CIRPointerPtrLikeModel
    : public mlir::PtrLikeTypeInterface::ExternalModel<CIRPointerPtrLikeModel,
                                                       cir::PointerType> {
  mlir::Attribute getMemorySpace(mlir::Type pointer) const {
    auto addrSpace = mlir::cast<cir::PointerType>(pointer).getAddrSpace();
    return addrSpace ? mlir::Attribute(addrSpace) : mlir::Attribute();
  }
  mlir::Type getElementType(mlir::Type pointer) const {
    return mlir::cast<cir::PointerType>(pointer).getPointee();
  }
  bool hasPtrMetadata(mlir::Type pointer) const { return false; }
  mlir::FailureOr<mlir::PtrLikeTypeInterface>
  clonePtrWith(mlir::Type pointer, mlir::Attribute memorySpace,
               std::optional<mlir::Type> elementType) const {
    auto ptrTy = mlir::cast<cir::PointerType>(pointer);
    mlir::Type eTy = elementType ? *elementType : ptrTy.getPointee();
    auto addrSpace =
        mlir::dyn_cast_or_null<mlir::ptr::MemorySpaceAttrInterface>(
            memorySpace);
    return mlir::cast<mlir::PtrLikeTypeInterface>(
        cir::PointerType::get(eTy, addrSpace));
  }
};
} // namespace

namespace cir::omp {

void registerOpenMPExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx, cir::CIRDialect *dialect) {
    cir::FuncOp::attachInterface<
        mlir::omp::DeclareTargetDefaultModel<cir::FuncOp>>(*ctx);
    cir::PointerType::attachInterface<CIRPointerPtrLikeModel>(*ctx);
  });
}

} // namespace cir::omp
