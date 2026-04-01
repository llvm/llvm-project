//===- FIROpenACCTypeInterfaces.h -------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains external dialect interfaces for FIR.
//
//===----------------------------------------------------------------------===//

#ifndef FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
#define FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_

#include "flang/Optimizer/Dialect/FIRType.h"
#include "aiir/Dialect/OpenACC/OpenACC.h"

namespace fir::acc {

template <typename T>
struct OpenACCPointerLikeModel
    : public aiir::acc::PointerLikeType::ExternalModel<
          OpenACCPointerLikeModel<T>, T> {
  aiir::Type getElementType(aiir::Type pointer) const {
    return aiir::cast<T>(pointer).getElementType();
  }
  aiir::acc::VariableTypeCategory
  getPointeeTypeCategory(aiir::Type pointer,
                         aiir::TypedValue<aiir::acc::PointerLikeType> varPtr,
                         aiir::Type varType) const;

  aiir::Value genAllocate(aiir::Type pointer, aiir::OpBuilder &builder,
                          aiir::Location loc, llvm::StringRef varName,
                          aiir::Type varType, aiir::Value originalVar,
                          bool &needsFree) const;

  bool genFree(aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
               aiir::TypedValue<aiir::acc::PointerLikeType> varToFree,
               aiir::Value allocRes, aiir::Type varType) const;

  bool genCopy(aiir::Type pointer, aiir::OpBuilder &builder, aiir::Location loc,
               aiir::TypedValue<aiir::acc::PointerLikeType> destination,
               aiir::TypedValue<aiir::acc::PointerLikeType> source,
               aiir::Type varType) const;

  aiir::Value genLoad(aiir::Type pointer, aiir::OpBuilder &builder,
                      aiir::Location loc,
                      aiir::TypedValue<aiir::acc::PointerLikeType> srcPtr,
                      aiir::Type valueType) const;

  bool genStore(aiir::Type pointer, aiir::OpBuilder &builder,
                aiir::Location loc, aiir::Value valueToStore,
                aiir::TypedValue<aiir::acc::PointerLikeType> destPtr) const;

  bool isDeviceData(aiir::Type pointer, aiir::Value var) const;
};

template <typename T>
struct OpenACCMappableModel
    : public aiir::acc::MappableType::ExternalModel<OpenACCMappableModel<T>,
                                                    T> {
  aiir::TypedValue<aiir::acc::PointerLikeType> getVarPtr(::aiir::Type type,
                                                         aiir::Value var) const;

  std::optional<llvm::TypeSize>
  getSizeInBytes(aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
                 const aiir::DataLayout &dataLayout) const;

  std::optional<int64_t>
  getOffsetInBytes(aiir::Type type, aiir::Value var, aiir::ValueRange accBounds,
                   const aiir::DataLayout &dataLayout) const;

  bool hasUnknownDimensions(aiir::Type type) const;

  llvm::SmallVector<aiir::Value>
  generateAccBounds(aiir::Type type, aiir::Value var,
                    aiir::OpBuilder &builder) const;

  aiir::acc::VariableTypeCategory getTypeCategory(aiir::Type type,
                                                  aiir::Value var) const;

  aiir::acc::VariableInfoAttr
  genPrivateVariableInfo(aiir::Type type,
                         aiir::TypedValue<aiir::acc::MappableType> var) const;

  aiir::Value generatePrivateInit(aiir::Type type, aiir::OpBuilder &builder,
                                  aiir::Location loc,
                                  aiir::TypedValue<aiir::acc::MappableType> var,
                                  llvm::StringRef varName,
                                  aiir::ValueRange extents, aiir::Value initVal,
                                  aiir::acc::VariableInfoAttr varInfo,
                                  bool &needsDestroy) const;

  bool generatePrivateDestroy(aiir::Type type, aiir::OpBuilder &builder,
                              aiir::Location loc, aiir::Value privatized,
                              aiir::ValueRange bounds,
                              aiir::acc::VariableInfoAttr varInfo) const;

  bool generateCopy(aiir::Type type, aiir::OpBuilder &aiirBuilder,
                    aiir::Location loc,
                    aiir::TypedValue<aiir::acc::MappableType> source,
                    aiir::TypedValue<aiir::acc::MappableType> dest,
                    aiir::ValueRange bounds,
                    aiir::acc::VariableInfoAttr varInfo) const;

  bool generateCombiner(aiir::Type type, aiir::OpBuilder &aiirBuilder,
                        aiir::Location loc,
                        aiir::TypedValue<aiir::acc::MappableType> dest,
                        aiir::TypedValue<aiir::acc::MappableType> source,
                        aiir::ValueRange bounds,
                        aiir::acc::ReductionOperator op,
                        aiir::Attribute fastmathFlags) const;

  bool isDeviceData(aiir::Type type, aiir::Value var) const;
};

struct OpenACCReducibleLogicalModel
    : public aiir::acc::ReducibleType::ExternalModel<
          OpenACCReducibleLogicalModel, fir::LogicalType> {
  std::optional<aiir::arith::AtomicRMWKind>
  getAtomicRMWKind(aiir::Type type, aiir::acc::ReductionOperator redOp) const;
};

} // namespace fir::acc

#endif // FLANG_OPTIMIZER_OPENACC_FIROPENACCTYPEINTERFACES_H_
