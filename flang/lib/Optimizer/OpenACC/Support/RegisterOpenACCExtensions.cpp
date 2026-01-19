//===-- RegisterOpenACCExtensions.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Registration for OpenACC extensions as applied to FIR dialect.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/OpenACC/Support/RegisterOpenACCExtensions.h"

#include "flang/Optimizer/Dialect/CUF/CUFDialect.h"
#include "flang/Optimizer/Dialect/CUF/CUFOps.h"
#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCOpsInterfaces.h"
#include "flang/Optimizer/OpenACC/Support/FIROpenACCTypeInterfaces.h"

namespace fir::acc {
void registerOpenACCExtensions(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            fir::FIROpsDialect *dialect) {
    fir::BoxType::attachInterface<OpenACCMappableModel<fir::BaseBoxType>>(*ctx);
    fir::ClassType::attachInterface<OpenACCMappableModel<fir::BaseBoxType>>(
        *ctx);
    fir::ReferenceType::attachInterface<
        OpenACCMappableModel<fir::ReferenceType>>(*ctx);
    fir::PointerType::attachInterface<OpenACCMappableModel<fir::PointerType>>(
        *ctx);
    fir::HeapType::attachInterface<OpenACCMappableModel<fir::HeapType>>(*ctx);

    fir::ReferenceType::attachInterface<
        OpenACCPointerLikeModel<fir::ReferenceType>>(*ctx);
    fir::PointerType::attachInterface<
        OpenACCPointerLikeModel<fir::PointerType>>(*ctx);
    fir::HeapType::attachInterface<OpenACCPointerLikeModel<fir::HeapType>>(
        *ctx);

    fir::LLVMPointerType::attachInterface<
        OpenACCPointerLikeModel<fir::LLVMPointerType>>(*ctx);

    fir::ArrayCoorOp::attachInterface<
        PartialEntityAccessModel<fir::ArrayCoorOp>>(*ctx);
    fir::CoordinateOp::attachInterface<
        PartialEntityAccessModel<fir::CoordinateOp>>(*ctx);
    fir::DeclareOp::attachInterface<PartialEntityAccessModel<fir::DeclareOp>>(
        *ctx);

    fir::AddrOfOp::attachInterface<AddressOfGlobalModel>(*ctx);
    fir::GlobalOp::attachInterface<GlobalVariableModel>(*ctx);

    fir::AllocaOp::attachInterface<IndirectGlobalAccessModel<fir::AllocaOp>>(
        *ctx);
    fir::EmboxOp::attachInterface<IndirectGlobalAccessModel<fir::EmboxOp>>(
        *ctx);
    fir::ReboxOp::attachInterface<IndirectGlobalAccessModel<fir::ReboxOp>>(
        *ctx);
    fir::TypeDescOp::attachInterface<
        IndirectGlobalAccessModel<fir::TypeDescOp>>(*ctx);

    // Attach OutlineRematerializationOpInterface to FIR operations that
    // produce synthetic types (shapes, field indices) which cannot be passed
    // as arguments to outlined regions and must be rematerialized inside.
    fir::ShapeOp::attachInterface<OutlineRematerializationModel<fir::ShapeOp>>(
        *ctx);
    fir::ShapeShiftOp::attachInterface<
        OutlineRematerializationModel<fir::ShapeShiftOp>>(*ctx);
    fir::ShiftOp::attachInterface<OutlineRematerializationModel<fir::ShiftOp>>(
        *ctx);
    fir::FieldIndexOp::attachInterface<
        OutlineRematerializationModel<fir::FieldIndexOp>>(*ctx);
  });

  // Register HLFIR operation interfaces
  registry.addExtension(
      +[](mlir::MLIRContext *ctx, hlfir::hlfirDialect *dialect) {
        hlfir::DesignateOp::attachInterface<
            PartialEntityAccessModel<hlfir::DesignateOp>>(*ctx);
        hlfir::DeclareOp::attachInterface<
            PartialEntityAccessModel<hlfir::DeclareOp>>(*ctx);
      });

  // Register CUF operation interfaces
  registry.addExtension(+[](mlir::MLIRContext *ctx, cuf::CUFDialect *dialect) {
    cuf::KernelOp::attachInterface<OffloadRegionModel<cuf::KernelOp>>(*ctx);
  });

  // Attach FIR dialect interfaces to OpenACC operations.
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::acc::OpenACCDialect *dialect) {
    mlir::acc::LoopOp::attachInterface<OperationMoveModel<mlir::acc::LoopOp>>(
        *ctx);
  });

  registerAttrsExtensions(registry);
}

} // namespace fir::acc
