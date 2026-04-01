//===- Tensor.cpp - C API for SparseTensor dialect ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/SparseTensor.h"
#include "aiir-c/IR.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/Dialect/SparseTensor/IR/SparseTensor.h"
#include "aiir/Support/LLVM.h"

using namespace llvm;
using namespace aiir::sparse_tensor;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(SparseTensor, sparse_tensor,
                                      aiir::sparse_tensor::SparseTensorDialect)

// Ensure the C-API enums are int-castable to C++ equivalents.
static_assert(
    static_cast<int>(AIIR_SPARSE_TENSOR_LEVEL_DENSE) ==
            static_cast<int>(LevelFormat::Dense) &&
        static_cast<int>(AIIR_SPARSE_TENSOR_LEVEL_COMPRESSED) ==
            static_cast<int>(LevelFormat::Compressed) &&
        static_cast<int>(AIIR_SPARSE_TENSOR_LEVEL_SINGLETON) ==
            static_cast<int>(LevelFormat::Singleton) &&
        static_cast<int>(AIIR_SPARSE_TENSOR_LEVEL_LOOSE_COMPRESSED) ==
            static_cast<int>(LevelFormat::LooseCompressed) &&
        static_cast<int>(AIIR_SPARSE_TENSOR_LEVEL_N_OUT_OF_M) ==
            static_cast<int>(LevelFormat::NOutOfM),
    "AiirSparseTensorLevelFormat (C-API) and LevelFormat (C++) mismatch");

static_assert(static_cast<int>(AIIR_SPARSE_PROPERTY_NON_ORDERED) ==
                      static_cast<int>(LevelPropNonDefault::Nonordered) &&
                  static_cast<int>(AIIR_SPARSE_PROPERTY_NON_UNIQUE) ==
                      static_cast<int>(LevelPropNonDefault::Nonunique) &&
                  static_cast<int>(AIIR_SPARSE_PROPERTY_SOA) ==
                      static_cast<int>(LevelPropNonDefault::SoA),
              "AiirSparseTensorLevelProperty (C-API) and "
              "LevelPropertyNondefault (C++) mismatch");

bool aiirAttributeIsASparseTensorEncodingAttr(AiirAttribute attr) {
  return isa<SparseTensorEncodingAttr>(unwrap(attr));
}

AiirAttribute aiirSparseTensorEncodingAttrGet(
    AiirContext ctx, intptr_t lvlRank,
    AiirSparseTensorLevelType const *lvlTypes, AiirAffineMap dimToLvl,
    AiirAffineMap lvlToDim, int posWidth, int crdWidth,
    AiirAttribute explicitVal, AiirAttribute implicitVal) {
  SmallVector<LevelType> cppLvlTypes;

  cppLvlTypes.reserve(lvlRank);
  for (intptr_t l = 0; l < lvlRank; ++l)
    cppLvlTypes.push_back(static_cast<LevelType>(lvlTypes[l]));

  return wrap(SparseTensorEncodingAttr::get(
      unwrap(ctx), cppLvlTypes, unwrap(dimToLvl), unwrap(lvlToDim), posWidth,
      crdWidth, unwrap(explicitVal), unwrap(implicitVal)));
}

AiirStringRef aiirSparseTensorEncodingAttrGetName(void) {
  return wrap(SparseTensorEncodingAttr::name);
}

AiirAffineMap aiirSparseTensorEncodingAttrGetDimToLvl(AiirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getDimToLvl());
}

AiirAffineMap aiirSparseTensorEncodingAttrGetLvlToDim(AiirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlToDim());
}

intptr_t aiirSparseTensorEncodingGetLvlRank(AiirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlRank();
}

AiirSparseTensorLevelType
aiirSparseTensorEncodingAttrGetLvlType(AiirAttribute attr, intptr_t lvl) {
  return static_cast<AiirSparseTensorLevelType>(
      cast<SparseTensorEncodingAttr>(unwrap(attr)).getLvlType(lvl));
}

enum AiirSparseTensorLevelFormat
aiirSparseTensorEncodingAttrGetLvlFmt(AiirAttribute attr, intptr_t lvl) {
  LevelType lt =
      static_cast<LevelType>(aiirSparseTensorEncodingAttrGetLvlType(attr, lvl));
  return static_cast<AiirSparseTensorLevelFormat>(lt.getLvlFmt());
}

int aiirSparseTensorEncodingAttrGetPosWidth(AiirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getPosWidth();
}

int aiirSparseTensorEncodingAttrGetCrdWidth(AiirAttribute attr) {
  return cast<SparseTensorEncodingAttr>(unwrap(attr)).getCrdWidth();
}

AiirAttribute aiirSparseTensorEncodingAttrGetExplicitVal(AiirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getExplicitVal());
}

AiirAttribute aiirSparseTensorEncodingAttrGetImplicitVal(AiirAttribute attr) {
  return wrap(cast<SparseTensorEncodingAttr>(unwrap(attr)).getImplicitVal());
}

AiirSparseTensorLevelType aiirSparseTensorEncodingAttrBuildLvlType(
    enum AiirSparseTensorLevelFormat lvlFmt,
    const enum AiirSparseTensorLevelPropertyNondefault *properties,
    unsigned size, unsigned n, unsigned m) {

  std::vector<LevelPropNonDefault> props;
  props.reserve(size);
  for (unsigned i = 0; i < size; i++)
    props.push_back(static_cast<LevelPropNonDefault>(properties[i]));

  return static_cast<AiirSparseTensorLevelType>(
      *buildLevelType(static_cast<LevelFormat>(lvlFmt), props, n, m));
}

unsigned
aiirSparseTensorEncodingAttrGetStructuredN(AiirSparseTensorLevelType lvlType) {
  return getN(static_cast<LevelType>(lvlType));
}

unsigned
aiirSparseTensorEncodingAttrGetStructuredM(AiirSparseTensorLevelType lvlType) {
  return getM(static_cast<LevelType>(lvlType));
}
