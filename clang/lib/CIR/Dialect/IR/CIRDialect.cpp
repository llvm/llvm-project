//===- CIRDialect.cpp - MLIR CIR ops implementation -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the CIR dialect. Its operations are implemented in the
// CIROps<Partition>.cpp files, one per op partition.
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/IR/CIRDialectBase.h"
#include "clang/CIR/Dialect/IR/CIROpsCore.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LLVM.h"

#include "clang/Basic/TargetCXXABI.h"
#include "clang/CIR/Dialect/IR/CIROpsDialect.cpp.inc"
#include "clang/CIR/Dialect/IR/CIROpsEnums.cpp.inc"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include "CIRDialectBytecode.h"

using namespace mlir;
using namespace cir;

//===----------------------------------------------------------------------===//
// CIR Dialect
//===----------------------------------------------------------------------===//
namespace {
struct CIROpAsmDialectInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Type type, raw_ostream &os) const final {
    if (auto recordType = dyn_cast<cir::RecordType>(type)) {
      StringAttr nameAttr = recordType.getName();
      if (!nameAttr)
        os << "rec_anon_" << recordType.getKindAsStr();
      else
        os << "rec_" << nameAttr.getValue();
      return AliasResult::OverridableAlias;
    }
    if (auto intType = dyn_cast<cir::IntType>(type)) {
      // We only provide alias for standard integer types (i.e. integer types
      // whose width is a power of 2 and at least 8).
      unsigned width = intType.getWidth();
      if (width < 8 || !llvm::isPowerOf2_32(width))
        return AliasResult::NoAlias;
      os << intType.getAlias();
      return AliasResult::OverridableAlias;
    }
    if (auto voidType = dyn_cast<cir::VoidType>(type)) {
      os << voidType.getAlias();
      return AliasResult::OverridableAlias;
    }

    return AliasResult::NoAlias;
  }

  AliasResult getAlias(Attribute attr, raw_ostream &os) const final {
    if (auto boolAttr = mlir::dyn_cast<cir::BoolAttr>(attr)) {
      os << (boolAttr.getValue() ? "true" : "false");
      return AliasResult::FinalAlias;
    }
    if (auto bitfield = mlir::dyn_cast<cir::BitfieldInfoAttr>(attr)) {
      os << "bfi_" << bitfield.getName().str();
      return AliasResult::FinalAlias;
    }
    if (auto dynCastInfoAttr = mlir::dyn_cast<cir::DynamicCastInfoAttr>(attr)) {
      os << dynCastInfoAttr.getAlias();
      return AliasResult::FinalAlias;
    }
    if (auto cmpThreeWayInfoAttr =
            mlir::dyn_cast<cir::CmpThreeWayInfoAttr>(attr)) {
      os << cmpThreeWayInfoAttr.getAlias();
      return AliasResult::FinalAlias;
    }
    return AliasResult::NoAlias;
  }
};
} // namespace

void cir::CIRDialect::initialize() {
  registerTypes();
  registerAttributes();
  registerOperations();
  addInterfaces<CIROpAsmDialectInterface>();
  detail::addBytecodeInterface(this);
}

Operation *cir::CIRDialect::materializeConstant(mlir::OpBuilder &builder,
                                                mlir::Attribute value,
                                                mlir::Type type,
                                                mlir::Location loc) {
  return cir::ConstantOp::create(builder, loc, type,
                                 mlir::cast<mlir::TypedAttr>(value));
}

static bool isOpenCLVersionAttrName(StringRef attrName) {
  return attrName == CIRDialect::getOpenCLVersionAttrName() ||
         attrName == CIRDialect::getOpenCLCXXVersionAttrName() ||
         attrName == CIRDialect::getOpenCLSPIRVersionAttrName();
}

static LogicalResult verifyOpenCLVersionAttrPlacement(Operation *op,
                                                      NamedAttribute attr) {
  StringRef attrName = attr.getName().getValue();
  if (isa<ModuleOp>(op))
    return success();

  return op->emitError() << attrName
                         << " attribute must be attached to a module";
}

static bool areOpenCLVersionsCompatible(cir::OpenCLVersionAttr openCLVersion,
                                        cir::OpenCLVersionAttr cxxVersion) {
  return (openCLVersion.getMajorVersion() == 2 &&
          openCLVersion.getMinorVersion() == 0 &&
          cxxVersion.getMajorVersion() == 1 &&
          cxxVersion.getMinorVersion() == 0) ||
         (openCLVersion.getMajorVersion() == 3 &&
          openCLVersion.getMinorVersion() == 0 &&
          cxxVersion.getMajorVersion() == 2021 &&
          cxxVersion.getMinorVersion() == 0);
}

static LogicalResult verifyOpenCLCXXVersion(ModuleOp module,
                                            cir::OpenCLVersionAttr cxxVersion) {
  Attribute openCLAttr =
      module->getAttr(CIRDialect::getOpenCLVersionAttrName());
  if (!openCLAttr)
    return module.emitError()
           << "module attribute '" << CIRDialect::getOpenCLCXXVersionAttrName()
           << "' requires the companion attribute '"
           << CIRDialect::getOpenCLVersionAttrName() << "'";

  auto openCLVersion = dyn_cast<cir::OpenCLVersionAttr>(openCLAttr);
  if (!openCLVersion)
    return success();

  if (!areOpenCLVersionsCompatible(openCLVersion, cxxVersion))
    return module.emitError("incompatible OpenCL and C++ for OpenCL versions");

  return success();
}

static LogicalResult verifyOpenCLVersionAttr(Operation *op,
                                             NamedAttribute attr) {
  if (failed(verifyOpenCLVersionAttrPlacement(op, attr)))
    return failure();

  StringRef attrName = attr.getName().getValue();
  auto version = dyn_cast<cir::OpenCLVersionAttr>(attr.getValue());
  if (!version) {
    return op->emitError() << "expected " << attrName
                           << " to be #cir.cl.version";
  }

  if (attrName == CIRDialect::getOpenCLCXXVersionAttrName())
    return verifyOpenCLCXXVersion(cast<ModuleOp>(op), version);

  return success();
}

LogicalResult cir::CIRDialect::verifyRegionArgAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned /*argIndex*/,
    NamedAttribute attr) {
  if (!isOpenCLVersionAttrName(attr.getName().getValue()))
    return success();

  return verifyOpenCLVersionAttrPlacement(op, attr);
}

LogicalResult cir::CIRDialect::verifyRegionResultAttribute(
    Operation *op, unsigned /*regionIndex*/, unsigned /*resultIndex*/,
    NamedAttribute attr) {
  if (!isOpenCLVersionAttrName(attr.getName().getValue()))
    return success();

  return verifyOpenCLVersionAttrPlacement(op, attr);
}

//===----------------------------------------------------------------------===//
// Dialect attribute verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyOffloadKind(mlir::ModuleOp module,
                                       cir::OffloadKind expected) {
  auto attr = module->getAttrOfType<cir::OffloadKindAttr>(
      cir::CIRDialect::getOffloadKindAttrName());
  if (!attr)
    return module.emitOpError()
           << "expects '" << cir::CIRDialect::getOffloadKindAttrName()
           << "' offload kind attribute";
  if (attr.getValue() != expected)
    return module.emitOpError()
           << "expects '" << cir::CIRDialect::getOffloadKindAttrName()
           << "' value '" << cir::stringifyOffloadKind(expected) << "'";
  return success();
}

// A module marked with `cir.offload.container` holds the host module followed
// by one or more device modules, each tagged with `cir.offload.kind`. Keeping
// the host module first gives later offload passes a simple convention for
// finding the host side while iterating the remaining device modules.
static LogicalResult verifyOffloadContainer(mlir::Operation *op) {
  auto container = mlir::dyn_cast<mlir::ModuleOp>(op);
  if (!container)
    return op->emitOpError()
           << "expects '" << cir::CIRDialect::getOffloadContainerAttrName()
           << "' attribute to be attached to '"
           << mlir::ModuleOp::getOperationName() << "'";

  mlir::Block &body = *container.getBody();
  if (body.empty())
    return container.emitOpError()
           << "expects host module as the first nested op";

  auto host = mlir::dyn_cast<mlir::ModuleOp>(body.front());
  if (!host)
    return container.emitOpError()
           << "expects host module as the first nested op";
  if (failed(verifyOffloadKind(host, cir::OffloadKind::Host)))
    return failure();

  // At least one device module is required after the host module.
  if (std::next(body.begin()) == body.end())
    return container.emitOpError() << "expects at least one device module";

  for (auto &op : llvm::drop_begin(body)) {
    auto module = mlir::dyn_cast<mlir::ModuleOp>(op);
    if (!module)
      return container.emitOpError()
             << "expects only nested builtin.module ops";
    if (failed(verifyOffloadKind(module, cir::OffloadKind::Device)))
      return failure();
  }
  return success();
}

LogicalResult
cir::CIRDialect::verifyOperationAttribute(mlir::Operation *op,
                                          mlir::NamedAttribute attr) {
  llvm::StringRef attrName = attr.getName().getValue();
  if (isOpenCLVersionAttrName(attrName))
    return verifyOpenCLVersionAttr(op, attr);

  if (attrName == getCXXABIAttrName()) {
    if (!mlir::isa<mlir::ModuleOp>(op))
      return op->emitOpError() << "expects '" << getCXXABIAttrName()
                               << "' attribute to be attached to '"
                               << mlir::ModuleOp::getOperationName() << "'";
    auto abi = mlir::dyn_cast<mlir::StringAttr>(attr.getValue());
    if (!abi)
      return op->emitOpError()
             << "expects '" << getCXXABIAttrName() << "' to be a string";

    StringRef abival = abi.getValue();
    if (!clang::TargetCXXABI::isABI(abival))
      return op->emitOpError() << "unknown C++ ABI '" << abi.getValue() << "'";

    clang::TargetCXXABI::Kind kind = clang::TargetCXXABI::getKind(abival);
    if (auto triple = op->getAttrOfType<mlir::StringAttr>(getTripleAttrName()))
      if (!clang::TargetCXXABI::isSupportedCXXABI(
              llvm::Triple(triple.getValue()), kind))
        return op->emitOpError()
               << "C++ ABI '" << abi.getValue()
               << "' is not supported on target '" << triple.getValue() << "'";

    return success();
  }

  if (attrName == getOffloadContainerAttrName()) {
    if (!mlir::isa<mlir::UnitAttr>(attr.getValue()))
      return op->emitOpError() << "expects '" << getOffloadContainerAttrName()
                               << "' to be a unit attribute";
    return verifyOffloadContainer(op);
  }

  // The container verifier owns the structural contract between a container
  // and the modules it holds. All this can add is that the kind attribute
  // never lands on something that is not a module.
  if (attrName == getOffloadKindAttrName() && !mlir::isa<mlir::ModuleOp>(op))
    return op->emitOpError() << "expects '" << getOffloadKindAttrName()
                             << "' attribute to be attached to '"
                             << mlir::ModuleOp::getOperationName() << "'";

  if (attrName == getModuleFramePointerAttrName()) {
    if (!mlir::isa<mlir::ModuleOp>(op))
      return op->emitOpError() << "expects '" << getModuleFramePointerAttrName()
                               << "' attribute to be attached to '"
                               << mlir::ModuleOp::getOperationName() << "'";
    if (!mlir::isa<cir::FramePointerKindAttr>(attr.getValue()))
      return op->emitOpError() << "expects '" << getModuleFramePointerAttrName()
                               << "' to be a #cir.frame_pointer attribute";
    return success();
  }

  // LoweringPrepare uses this attribute directly as the fatbin global's
  // initializer, so it must be a valid #cir.const_array payload for its type.
  if (attrName == getCUDADeviceBinaryAttrName()) {
    auto bytes = mlir::dyn_cast<mlir::StringAttr>(attr.getValue());
    auto arrayTy =
        bytes ? mlir::dyn_cast<cir::ArrayType>(bytes.getType()) : nullptr;
    if (!arrayTy || arrayTy.getSize() != bytes.size())
      return op->emitOpError()
             << "expects '" << getCUDADeviceBinaryAttrName()
             << "' to be a string typed as an array of its length";
    return cir::ConstArrayAttr::verify([&] { return op->emitOpError(); },
                                       arrayTy, bytes,
                                       /*trailingZerosNum=*/0);
  }

  return success();
}
