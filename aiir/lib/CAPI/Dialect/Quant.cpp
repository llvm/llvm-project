//===- Quant.cpp - C Interface for Quant dialect --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/Quant.h"
#include "aiir-c/BuiltinAttributes.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/Quant/IR/Quant.h"
#include "aiir/Dialect/Quant/IR/QuantTypes.h"

using namespace aiir;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(quant, quant, quant::QuantDialect)

//===---------------------------------------------------------------------===//
// QuantizedType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAQuantizedType(AiirType type) {
  return isa<quant::QuantizedType>(unwrap(type));
}

unsigned aiirQuantizedTypeGetSignedFlag() {
  return quant::QuantizationFlags::Signed;
}

int64_t aiirQuantizedTypeGetDefaultMinimumForInteger(bool isSigned,
                                                     unsigned integralWidth) {
  return quant::QuantizedType::getDefaultMinimumForInteger(isSigned,
                                                           integralWidth);
}

int64_t aiirQuantizedTypeGetDefaultMaximumForInteger(bool isSigned,
                                                     unsigned integralWidth) {
  return quant::QuantizedType::getDefaultMaximumForInteger(isSigned,
                                                           integralWidth);
}

AiirType aiirQuantizedTypeGetExpressedType(AiirType type) {
  return wrap(cast<quant::QuantizedType>(unwrap(type)).getExpressedType());
}

unsigned aiirQuantizedTypeGetFlags(AiirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getFlags();
}

bool aiirQuantizedTypeIsSigned(AiirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).isSigned();
}

AiirType aiirQuantizedTypeGetStorageType(AiirType type) {
  return wrap(cast<quant::QuantizedType>(unwrap(type)).getStorageType());
}

int64_t aiirQuantizedTypeGetStorageTypeMin(AiirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeMin();
}

int64_t aiirQuantizedTypeGetStorageTypeMax(AiirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeMax();
}

unsigned aiirQuantizedTypeGetStorageTypeIntegralWidth(AiirType type) {
  return cast<quant::QuantizedType>(unwrap(type)).getStorageTypeIntegralWidth();
}

bool aiirQuantizedTypeIsCompatibleExpressedType(AiirType type,
                                                AiirType candidate) {
  return cast<quant::QuantizedType>(unwrap(type))
      .isCompatibleExpressedType(unwrap(candidate));
}

AiirType aiirQuantizedTypeGetQuantizedElementType(AiirType type) {
  return wrap(quant::QuantizedType::getQuantizedElementType(unwrap(type)));
}

AiirType aiirQuantizedTypeCastFromStorageType(AiirType type,
                                              AiirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castFromStorageType(unwrap(candidate)));
}

AiirType aiirQuantizedTypeCastToStorageType(AiirType type) {
  return wrap(quant::QuantizedType::castToStorageType(
      cast<quant::QuantizedType>(unwrap(type))));
}

AiirType aiirQuantizedTypeCastFromExpressedType(AiirType type,
                                                AiirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castFromExpressedType(unwrap(candidate)));
}

AiirType aiirQuantizedTypeCastToExpressedType(AiirType type) {
  return wrap(quant::QuantizedType::castToExpressedType(unwrap(type)));
}

AiirType aiirQuantizedTypeCastExpressedToStorageType(AiirType type,
                                                     AiirType candidate) {
  return wrap(cast<quant::QuantizedType>(unwrap(type))
                  .castExpressedToStorageType(unwrap(candidate)));
}

//===---------------------------------------------------------------------===//
// AnyQuantizedType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAAnyQuantizedType(AiirType type) {
  return isa<quant::AnyQuantizedType>(unwrap(type));
}

AiirTypeID aiirAnyQuantizedTypeGetTypeID(void) {
  return wrap(quant::AnyQuantizedType::getTypeID());
}

AiirType aiirAnyQuantizedTypeGet(unsigned flags, AiirType storageType,
                                 AiirType expressedType, int64_t storageTypeMin,
                                 int64_t storageTypeMax) {
  return wrap(quant::AnyQuantizedType::get(flags, unwrap(storageType),
                                           unwrap(expressedType),
                                           storageTypeMin, storageTypeMax));
}

AiirStringRef aiirAnyQuantizedTypeGetName(void) {
  return wrap(quant::AnyQuantizedType::name);
}

//===---------------------------------------------------------------------===//
// UniformQuantizedType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAUniformQuantizedType(AiirType type) {
  return isa<quant::UniformQuantizedType>(unwrap(type));
}

AiirTypeID aiirUniformQuantizedTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedType::getTypeID());
}

AiirType aiirUniformQuantizedTypeGet(unsigned flags, AiirType storageType,
                                     AiirType expressedType, double scale,
                                     int64_t zeroPoint, int64_t storageTypeMin,
                                     int64_t storageTypeMax) {
  return wrap(quant::UniformQuantizedType::get(
      flags, unwrap(storageType), unwrap(expressedType), scale, zeroPoint,
      storageTypeMin, storageTypeMax));
}

AiirStringRef aiirUniformQuantizedTypeGetName(void) {
  return wrap(quant::UniformQuantizedType::name);
}

double aiirUniformQuantizedTypeGetScale(AiirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).getScale();
}

int64_t aiirUniformQuantizedTypeGetZeroPoint(AiirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).getZeroPoint();
}

bool aiirUniformQuantizedTypeIsFixedPoint(AiirType type) {
  return cast<quant::UniformQuantizedType>(unwrap(type)).isFixedPoint();
}

//===---------------------------------------------------------------------===//
// UniformQuantizedPerAxisType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAUniformQuantizedPerAxisType(AiirType type) {
  return isa<quant::UniformQuantizedPerAxisType>(unwrap(type));
}

AiirTypeID aiirUniformQuantizedPerAxisTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedPerAxisType::getTypeID());
}

AiirType aiirUniformQuantizedPerAxisTypeGet(
    unsigned flags, AiirType storageType, AiirType expressedType,
    intptr_t nDims, double *scales, int64_t *zeroPoints,
    int32_t quantizedDimension, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  return wrap(quant::UniformQuantizedPerAxisType::get(
      flags, unwrap(storageType), unwrap(expressedType),
      llvm::ArrayRef(scales, nDims), llvm::ArrayRef(zeroPoints, nDims),
      quantizedDimension, storageTypeMin, storageTypeMax));
}

AiirStringRef aiirUniformQuantizedPerAxisTypeGetName(void) {
  return wrap(quant::UniformQuantizedPerAxisType::name);
}

intptr_t aiirUniformQuantizedPerAxisTypeGetNumDims(AiirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getScales()
      .size();
}

double aiirUniformQuantizedPerAxisTypeGetScale(AiirType type, intptr_t pos) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getScales()[pos];
}

int64_t aiirUniformQuantizedPerAxisTypeGetZeroPoint(AiirType type,
                                                    intptr_t pos) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getZeroPoints()[pos];
}

int32_t aiirUniformQuantizedPerAxisTypeGetQuantizedDimension(AiirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type))
      .getQuantizedDimension();
}

bool aiirUniformQuantizedPerAxisTypeIsFixedPoint(AiirType type) {
  return cast<quant::UniformQuantizedPerAxisType>(unwrap(type)).isFixedPoint();
}

//===---------------------------------------------------------------------===//
// UniformQuantizedSubChannelType
//===---------------------------------------------------------------------===//

bool aiirTypeIsAUniformQuantizedSubChannelType(AiirType type) {
  return isa<quant::UniformQuantizedSubChannelType>(unwrap(type));
}

AiirTypeID aiirUniformQuantizedSubChannelTypeGetTypeID(void) {
  return wrap(quant::UniformQuantizedSubChannelType::getTypeID());
}

AiirType aiirUniformQuantizedSubChannelTypeGet(
    unsigned flags, AiirType storageType, AiirType expressedType,
    AiirAttribute scalesAttr, AiirAttribute zeroPointsAttr, intptr_t nDims,
    int32_t *quantizedDimensions, int64_t *blockSizes, int64_t storageTypeMin,
    int64_t storageTypeMax) {
  auto scales = dyn_cast<aiir::DenseElementsAttr>(unwrap(scalesAttr));
  auto zeroPoints = dyn_cast<aiir::DenseElementsAttr>(unwrap(zeroPointsAttr));

  if (!scales || !zeroPoints) {
    return {};
  }

  return wrap(quant::UniformQuantizedSubChannelType::get(
      flags, unwrap(storageType), unwrap(expressedType), scales, zeroPoints,
      llvm::ArrayRef<int32_t>(quantizedDimensions, nDims),
      llvm::ArrayRef<int64_t>(blockSizes, nDims), storageTypeMin,
      storageTypeMax));
}

AiirStringRef aiirUniformQuantizedSubChannelTypeGetName(void) {
  return wrap(quant::UniformQuantizedSubChannelType::name);
}

intptr_t aiirUniformQuantizedSubChannelTypeGetNumBlockSizes(AiirType type) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getBlockSizes()
      .size();
}

int32_t aiirUniformQuantizedSubChannelTypeGetQuantizedDimension(AiirType type,
                                                                intptr_t pos) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getQuantizedDimensions()[pos];
}

int64_t aiirUniformQuantizedSubChannelTypeGetBlockSize(AiirType type,
                                                       intptr_t pos) {
  return cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
      .getBlockSizes()[pos];
}

AiirAttribute aiirUniformQuantizedSubChannelTypeGetScales(AiirType type) {
  return wrap(
      cast<quant::UniformQuantizedSubChannelType>(unwrap(type)).getScales());
}

AiirAttribute aiirUniformQuantizedSubChannelTypeGetZeroPoints(AiirType type) {
  return wrap(cast<quant::UniformQuantizedSubChannelType>(unwrap(type))
                  .getZeroPoints());
}

//===---------------------------------------------------------------------===//
// CalibratedQuantizedType
//===---------------------------------------------------------------------===//

bool aiirTypeIsACalibratedQuantizedType(AiirType type) {
  return isa<quant::CalibratedQuantizedType>(unwrap(type));
}

AiirTypeID aiirCalibratedQuantizedTypeGetTypeID(void) {
  return wrap(quant::CalibratedQuantizedType::getTypeID());
}

AiirType aiirCalibratedQuantizedTypeGet(AiirType expressedType, double min,
                                        double max) {
  return wrap(
      quant::CalibratedQuantizedType::get(unwrap(expressedType), min, max));
}

AiirStringRef aiirCalibratedQuantizedTypeGetName(void) {
  return wrap(quant::CalibratedQuantizedType::name);
}

double aiirCalibratedQuantizedTypeGetMin(AiirType type) {
  return cast<quant::CalibratedQuantizedType>(unwrap(type)).getMin();
}

double aiirCalibratedQuantizedTypeGetMax(AiirType type) {
  return cast<quant::CalibratedQuantizedType>(unwrap(type)).getMax();
}
