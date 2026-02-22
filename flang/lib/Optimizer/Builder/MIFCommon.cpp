//===-- MIFCommon.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/MIFCommon.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Dialect/MIF/MIFOps.h"
#include "flang/Optimizer/Dialect/Support/KindMapping.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/ADT/TypeSwitch.h"

std::string mif::getFullUniqName(mlir::Value addr) {
  mlir::Operation *op = addr.getDefiningOp();
  if (auto designateOp = mlir::dyn_cast<hlfir::DesignateOp>(op)) {
    if (designateOp.getComponent())
      return getFullUniqName(designateOp.getMemref()) + "." +
             designateOp.getComponent()->getValue().str();
    return getFullUniqName(designateOp.getMemref());
  } else if (auto declareOp = mlir::dyn_cast<hlfir::DeclareOp>(op))
    return declareOp.getUniqName().getValue().str();
  else if (auto declareOp = mlir::dyn_cast<fir::DeclareOp>(op))
    return declareOp.getUniqName().getValue().str();
  else if (auto load = mlir::dyn_cast<fir::LoadOp>(op))
    return getFullUniqName(load.getMemref());
  else if (auto ba = mlir::dyn_cast<fir::BoxAddrOp>(op))
    return getFullUniqName(ba.getVal());
  else if (auto rb = mlir::dyn_cast<fir::ReboxOp>(op))
    return getFullUniqName(rb.getBox());
  else if (auto eb = mlir::dyn_cast<fir::EmboxOp>(op))
    return getFullUniqName(eb.getMemref());
  else if (auto ebc = mlir::dyn_cast<fir::EmboxCharOp>(op))
    return getFullUniqName(ebc.getMemref());
  else if (auto c = mlir::dyn_cast<fir::CoordinateOp>(op)) {
    if (c.getFieldIndicesAttr()) {
      mlir::Type eleTy = fir::getFortranElementType(c.getRef().getType());
      std::string uniqName = getFullUniqName(c.getRef());
      for (auto index : c.getIndices()) {
        llvm::TypeSwitch<fir::IntOrValue>(index)
            .Case<mlir::IntegerAttr>([&](mlir::IntegerAttr intAttr) {
              if (auto recordType = llvm::dyn_cast<fir::RecordType>(eleTy)) {
                int fieldId = intAttr.getInt();
                if (fieldId < static_cast<int>(recordType.getNumFields())) {
                  auto nameAndType = recordType.getTypeList()[fieldId];
                  auto rrr = getFullUniqName(c.getRef()) + "." +
                             std::get<std::string>(nameAndType);
                  uniqName += "." + std::get<std::string>(nameAndType);
                }
              }
            })
            .Case<mlir::Value>(
                [&](mlir::Value v) { return getFullUniqName(v); });
      }
      return uniqName;
    }
    return getFullUniqName(c.getRef());
  }
  return "";
}
