//===- BuiltinAttributes.cpp - C Interface to AIIR Builtin Attributes -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/BuiltinAttributes.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/IntegerSet.h"
#include "aiir/CAPI/Support.h"
#include "aiir/IR/AsmState.h"
#include "aiir/IR/Attributes.h"
#include "aiir/IR/BuiltinAttributes.h"
#include "aiir/IR/BuiltinTypes.h"

using namespace aiir;

AiirAttribute aiirAttributeGetNull() { return {nullptr}; }

//===----------------------------------------------------------------------===//
// Location attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsALocation(AiirAttribute attr) {
  return llvm::isa<LocationAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAAffineMap(AiirAttribute attr) {
  return llvm::isa<AffineMapAttr>(unwrap(attr));
}

AiirAttribute aiirAffineMapAttrGet(AiirAffineMap map) {
  return wrap(AffineMapAttr::get(unwrap(map)));
}

AiirStringRef aiirAffineMapAttrGetName(void) {
  return wrap(AffineMapAttr::name);
}

AiirAffineMap aiirAffineMapAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<AffineMapAttr>(unwrap(attr)).getValue());
}

AiirTypeID aiirAffineMapAttrGetTypeID(void) {
  return wrap(AffineMapAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAArray(AiirAttribute attr) {
  return llvm::isa<ArrayAttr>(unwrap(attr));
}

AiirAttribute aiirArrayAttrGet(AiirContext ctx, intptr_t numElements,
                               AiirAttribute const *elements) {
  SmallVector<Attribute, 8> attrs;
  return wrap(
      ArrayAttr::get(unwrap(ctx), unwrapList(static_cast<size_t>(numElements),
                                             elements, attrs)));
}

AiirStringRef aiirArrayAttrGetName(void) { return wrap(ArrayAttr::name); }

intptr_t aiirArrayAttrGetNumElements(AiirAttribute attr) {
  return static_cast<intptr_t>(llvm::cast<ArrayAttr>(unwrap(attr)).size());
}

AiirAttribute aiirArrayAttrGetElement(AiirAttribute attr, intptr_t pos) {
  return wrap(llvm::cast<ArrayAttr>(unwrap(attr)).getValue()[pos]);
}

AiirTypeID aiirArrayAttrGetTypeID(void) { return wrap(ArrayAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsADictionary(AiirAttribute attr) {
  return llvm::isa<DictionaryAttr>(unwrap(attr));
}

AiirAttribute aiirDictionaryAttrGet(AiirContext ctx, intptr_t numElements,
                                    AiirNamedAttribute const *elements) {
  SmallVector<NamedAttribute, 8> attributes;
  attributes.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    attributes.emplace_back(unwrap(elements[i].name),
                            unwrap(elements[i].attribute));
  return wrap(DictionaryAttr::get(unwrap(ctx), attributes));
}

AiirStringRef aiirDictionaryAttrGetName(void) {
  return wrap(DictionaryAttr::name);
}

intptr_t aiirDictionaryAttrGetNumElements(AiirAttribute attr) {
  return static_cast<intptr_t>(llvm::cast<DictionaryAttr>(unwrap(attr)).size());
}

AiirNamedAttribute aiirDictionaryAttrGetElement(AiirAttribute attr,
                                                intptr_t pos) {
  NamedAttribute attribute =
      llvm::cast<DictionaryAttr>(unwrap(attr)).getValue()[pos];
  return {wrap(attribute.getName()), wrap(attribute.getValue())};
}

AiirAttribute aiirDictionaryAttrGetElementByName(AiirAttribute attr,
                                                 AiirStringRef name) {
  return wrap(llvm::cast<DictionaryAttr>(unwrap(attr)).get(unwrap(name)));
}

AiirTypeID aiirDictionaryAttrGetTypeID(void) {
  return wrap(DictionaryAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAFloat(AiirAttribute attr) {
  return llvm::isa<FloatAttr>(unwrap(attr));
}

AiirStringRef aiirFloatAttrGetName(void) { return wrap(FloatAttr::name); }

AiirAttribute aiirFloatAttrDoubleGet(AiirContext ctx, AiirType type,
                                     double value) {
  return wrap(FloatAttr::get(unwrap(type), value));
}

AiirAttribute aiirFloatAttrDoubleGetChecked(AiirLocation loc, AiirType type,
                                            double value) {
  return wrap(FloatAttr::getChecked(unwrap(loc), unwrap(type), value));
}

double aiirFloatAttrGetValueDouble(AiirAttribute attr) {
  return llvm::cast<FloatAttr>(unwrap(attr)).getValueAsDouble();
}

AiirTypeID aiirFloatAttrGetTypeID(void) { return wrap(FloatAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAInteger(AiirAttribute attr) {
  return llvm::isa<IntegerAttr>(unwrap(attr));
}

AiirAttribute aiirIntegerAttrGet(AiirType type, int64_t value) {
  return wrap(IntegerAttr::get(unwrap(type), value));
}

AiirStringRef aiirIntegerAttrGetName(void) { return wrap(IntegerAttr::name); }

int64_t aiirIntegerAttrGetValueInt(AiirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getInt();
}

int64_t aiirIntegerAttrGetValueSInt(AiirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getSInt();
}

uint64_t aiirIntegerAttrGetValueUInt(AiirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getUInt();
}

unsigned aiirIntegerAttrGetValueBitWidth(AiirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getValue().getBitWidth();
}

unsigned aiirIntegerAttrGetValueNumWords(AiirAttribute attr) {
  return llvm::cast<IntegerAttr>(unwrap(attr)).getValue().getNumWords();
}

void aiirIntegerAttrGetValueWords(AiirAttribute attr, uint64_t *words) {
  const APInt &value = llvm::cast<IntegerAttr>(unwrap(attr)).getValue();
  unsigned numWords = value.getNumWords();
  const uint64_t *rawData = value.getRawData();
  std::copy(rawData, rawData + numWords, words);
}

AiirAttribute aiirIntegerAttrGetFromWords(AiirType type, unsigned numWords,
                                          const uint64_t *words) {
  Type aiirType = unwrap(type);
  unsigned bitWidth = aiirType.getIntOrFloatBitWidth();
  APInt value(bitWidth, ArrayRef<uint64_t>(words, numWords));
  return wrap(IntegerAttr::get(aiirType, value));
}

AiirTypeID aiirIntegerAttrGetTypeID(void) {
  return wrap(IntegerAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsABool(AiirAttribute attr) {
  return llvm::isa<BoolAttr>(unwrap(attr));
}

AiirAttribute aiirBoolAttrGet(AiirContext ctx, int value) {
  return wrap(BoolAttr::get(unwrap(ctx), value));
}

bool aiirBoolAttrGetValue(AiirAttribute attr) {
  return llvm::cast<BoolAttr>(unwrap(attr)).getValue();
}

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAIntegerSet(AiirAttribute attr) {
  return llvm::isa<IntegerSetAttr>(unwrap(attr));
}

AiirTypeID aiirIntegerSetAttrGetTypeID(void) {
  return wrap(IntegerSetAttr::getTypeID());
}

AiirAttribute aiirIntegerSetAttrGet(AiirIntegerSet set) {
  return wrap(IntegerSetAttr::get(unwrap(set)));
}

AiirStringRef aiirIntegerSetAttrGetName(void) {
  return wrap(IntegerSetAttr::name);
}

AiirIntegerSet aiirIntegerSetAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<IntegerSetAttr>(unwrap(attr)).getValue());
}

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAOpaque(AiirAttribute attr) {
  return llvm::isa<OpaqueAttr>(unwrap(attr));
}

AiirAttribute aiirOpaqueAttrGet(AiirContext ctx, AiirStringRef dialectNamespace,
                                intptr_t dataLength, const char *data,
                                AiirType type) {
  return wrap(
      OpaqueAttr::get(StringAttr::get(unwrap(ctx), unwrap(dialectNamespace)),
                      StringRef(data, dataLength), unwrap(type)));
}

AiirStringRef aiirOpaqueAttrGetName(void) { return wrap(OpaqueAttr::name); }

AiirStringRef aiirOpaqueAttrGetDialectNamespace(AiirAttribute attr) {
  return wrap(
      llvm::cast<OpaqueAttr>(unwrap(attr)).getDialectNamespace().strref());
}

AiirStringRef aiirOpaqueAttrGetData(AiirAttribute attr) {
  return wrap(llvm::cast<OpaqueAttr>(unwrap(attr)).getAttrData());
}

AiirTypeID aiirOpaqueAttrGetTypeID(void) {
  return wrap(OpaqueAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAString(AiirAttribute attr) {
  return llvm::isa<StringAttr>(unwrap(attr));
}

AiirAttribute aiirStringAttrGet(AiirContext ctx, AiirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(ctx), unwrap(str)));
}

AiirStringRef aiirStringAttrGetName(void) { return wrap(StringAttr::name); }

AiirAttribute aiirStringAttrTypedGet(AiirType type, AiirStringRef str) {
  return wrap((Attribute)StringAttr::get(unwrap(str), unwrap(type)));
}

AiirStringRef aiirStringAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<StringAttr>(unwrap(attr)).getValue());
}

AiirTypeID aiirStringAttrGetTypeID(void) {
  return wrap(StringAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsASymbolRef(AiirAttribute attr) {
  return llvm::isa<SymbolRefAttr>(unwrap(attr));
}

AiirAttribute aiirSymbolRefAttrGet(AiirContext ctx, AiirStringRef symbol,
                                   intptr_t numReferences,
                                   AiirAttribute const *references) {
  SmallVector<FlatSymbolRefAttr, 4> refs;
  refs.reserve(numReferences);
  for (intptr_t i = 0; i < numReferences; ++i)
    refs.push_back(llvm::cast<FlatSymbolRefAttr>(unwrap(references[i])));
  auto symbolAttr = StringAttr::get(unwrap(ctx), unwrap(symbol));
  return wrap(SymbolRefAttr::get(symbolAttr, refs));
}

AiirStringRef aiirSymbolRefAttrGetName(void) {
  return wrap(SymbolRefAttr::name);
}

AiirStringRef aiirSymbolRefAttrGetRootReference(AiirAttribute attr) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getRootReference().getValue());
}

AiirStringRef aiirSymbolRefAttrGetLeafReference(AiirAttribute attr) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getLeafReference().getValue());
}

intptr_t aiirSymbolRefAttrGetNumNestedReferences(AiirAttribute attr) {
  return static_cast<intptr_t>(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getNestedReferences().size());
}

AiirAttribute aiirSymbolRefAttrGetNestedReference(AiirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      llvm::cast<SymbolRefAttr>(unwrap(attr)).getNestedReferences()[pos]);
}

AiirTypeID aiirSymbolRefAttrGetTypeID(void) {
  return wrap(SymbolRefAttr::getTypeID());
}

AiirAttribute aiirDistinctAttrCreate(AiirAttribute referencedAttr) {
  return wrap(aiir::DistinctAttr::create(unwrap(referencedAttr)));
}

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAFlatSymbolRef(AiirAttribute attr) {
  return llvm::isa<FlatSymbolRefAttr>(unwrap(attr));
}

AiirAttribute aiirFlatSymbolRefAttrGet(AiirContext ctx, AiirStringRef symbol) {
  return wrap(FlatSymbolRefAttr::get(unwrap(ctx), unwrap(symbol)));
}

AiirStringRef aiirFlatSymbolRefAttrGetName(void) {
  return wrap(FlatSymbolRefAttr::name);
}

AiirStringRef aiirFlatSymbolRefAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<FlatSymbolRefAttr>(unwrap(attr)).getValue());
}

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAType(AiirAttribute attr) {
  return llvm::isa<TypeAttr>(unwrap(attr));
}

AiirAttribute aiirTypeAttrGet(AiirType type) {
  return wrap(TypeAttr::get(unwrap(type)));
}

AiirStringRef aiirTypeAttrGetName(void) { return wrap(TypeAttr::name); }

AiirType aiirTypeAttrGetValue(AiirAttribute attr) {
  return wrap(llvm::cast<TypeAttr>(unwrap(attr)).getValue());
}

AiirTypeID aiirTypeAttrGetTypeID(void) { return wrap(TypeAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAUnit(AiirAttribute attr) {
  return llvm::isa<UnitAttr>(unwrap(attr));
}

AiirAttribute aiirUnitAttrGet(AiirContext ctx) {
  return wrap(UnitAttr::get(unwrap(ctx)));
}

AiirStringRef aiirUnitAttrGetName(void) { return wrap(UnitAttr::name); }

AiirTypeID aiirUnitAttrGetTypeID(void) { return wrap(UnitAttr::getTypeID()); }

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAElements(AiirAttribute attr) {
  return llvm::isa<ElementsAttr>(unwrap(attr));
}

AiirAttribute aiirElementsAttrGetValue(AiirAttribute attr, intptr_t rank,
                                       uint64_t *idxs) {
  return wrap(llvm::cast<ElementsAttr>(unwrap(attr))
                  .getValues<Attribute>()[llvm::ArrayRef(idxs, rank)]);
}

bool aiirElementsAttrIsValidIndex(AiirAttribute attr, intptr_t rank,
                                  uint64_t *idxs) {
  return llvm::cast<ElementsAttr>(unwrap(attr))
      .isValidIndex(llvm::ArrayRef(idxs, rank));
}

int64_t aiirElementsAttrGetNumElements(AiirAttribute attr) {
  return llvm::cast<ElementsAttr>(unwrap(attr)).getNumElements();
}

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//

AiirTypeID aiirDenseArrayAttrGetTypeID() {
  return wrap(DenseArrayAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// IsA support.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsADenseBoolArray(AiirAttribute attr) {
  return llvm::isa<DenseBoolArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseI8Array(AiirAttribute attr) {
  return llvm::isa<DenseI8ArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseI16Array(AiirAttribute attr) {
  return llvm::isa<DenseI16ArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseI32Array(AiirAttribute attr) {
  return llvm::isa<DenseI32ArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseI64Array(AiirAttribute attr) {
  return llvm::isa<DenseI64ArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseF32Array(AiirAttribute attr) {
  return llvm::isa<DenseF32ArrayAttr>(unwrap(attr));
}
bool aiirAttributeIsADenseF64Array(AiirAttribute attr) {
  return llvm::isa<DenseF64ArrayAttr>(unwrap(attr));
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

AiirAttribute aiirDenseBoolArrayGet(AiirContext ctx, intptr_t size,
                                    int const *values) {
  SmallVector<bool, 4> elements(values, values + size);
  return wrap(DenseBoolArrayAttr::get(unwrap(ctx), elements));
}
AiirAttribute aiirDenseI8ArrayGet(AiirContext ctx, intptr_t size,
                                  int8_t const *values) {
  return wrap(
      DenseI8ArrayAttr::get(unwrap(ctx), ArrayRef<int8_t>(values, size)));
}
AiirAttribute aiirDenseI16ArrayGet(AiirContext ctx, intptr_t size,
                                   int16_t const *values) {
  return wrap(
      DenseI16ArrayAttr::get(unwrap(ctx), ArrayRef<int16_t>(values, size)));
}
AiirAttribute aiirDenseI32ArrayGet(AiirContext ctx, intptr_t size,
                                   int32_t const *values) {
  return wrap(
      DenseI32ArrayAttr::get(unwrap(ctx), ArrayRef<int32_t>(values, size)));
}
AiirAttribute aiirDenseI64ArrayGet(AiirContext ctx, intptr_t size,
                                   int64_t const *values) {
  return wrap(
      DenseI64ArrayAttr::get(unwrap(ctx), ArrayRef<int64_t>(values, size)));
}
AiirAttribute aiirDenseF32ArrayGet(AiirContext ctx, intptr_t size,
                                   float const *values) {
  return wrap(
      DenseF32ArrayAttr::get(unwrap(ctx), ArrayRef<float>(values, size)));
}
AiirAttribute aiirDenseF64ArrayGet(AiirContext ctx, intptr_t size,
                                   double const *values) {
  return wrap(
      DenseF64ArrayAttr::get(unwrap(ctx), ArrayRef<double>(values, size)));
}

//===----------------------------------------------------------------------===//
// Accessors.
//===----------------------------------------------------------------------===//

intptr_t aiirDenseArrayGetNumElements(AiirAttribute attr) {
  return llvm::cast<DenseArrayAttr>(unwrap(attr)).size();
}

//===----------------------------------------------------------------------===//
// Indexed accessors.
//===----------------------------------------------------------------------===//

bool aiirDenseBoolArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseBoolArrayAttr>(unwrap(attr))[pos];
}
int8_t aiirDenseI8ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI8ArrayAttr>(unwrap(attr))[pos];
}
int16_t aiirDenseI16ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI16ArrayAttr>(unwrap(attr))[pos];
}
int32_t aiirDenseI32ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI32ArrayAttr>(unwrap(attr))[pos];
}
int64_t aiirDenseI64ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseI64ArrayAttr>(unwrap(attr))[pos];
}
float aiirDenseF32ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseF32ArrayAttr>(unwrap(attr))[pos];
}
double aiirDenseF64ArrayGetElement(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseF64ArrayAttr>(unwrap(attr))[pos];
}

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// IsA support.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsADenseElements(AiirAttribute attr) {
  return llvm::isa<DenseElementsAttr>(unwrap(attr));
}

bool aiirAttributeIsADenseIntElements(AiirAttribute attr) {
  return llvm::isa<DenseIntElementsAttr>(unwrap(attr));
}

bool aiirAttributeIsADenseFPElements(AiirAttribute attr) {
  return llvm::isa<DenseFPElementsAttr>(unwrap(attr));
}

AiirTypeID aiirDenseTypedElementsAttrGetTypeID(void) {
  return wrap(DenseTypedElementsAttr::getTypeID());
}

// Deprecated API. Will be removed in the future.
AiirTypeID aiirDenseIntOrFPElementsAttrGetTypeID(void) {
  return aiirDenseTypedElementsAttrGetTypeID();
}

//===----------------------------------------------------------------------===//
// Constructors.
//===----------------------------------------------------------------------===//

AiirAttribute aiirDenseElementsAttrGet(AiirType shapedType,
                                       intptr_t numElements,
                                       AiirAttribute const *elements) {
  SmallVector<Attribute, 8> attributes;
  return wrap(
      DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                             unwrapList(numElements, elements, attributes)));
}

AiirAttribute aiirDenseElementsAttrRawBufferGet(AiirType shapedType,
                                                size_t rawBufferSize,
                                                const void *rawBuffer) {
  auto shapedTypeCpp = llvm::cast<ShapedType>(unwrap(shapedType));
  ArrayRef<char> rawBufferCpp(static_cast<const char *>(rawBuffer),
                              rawBufferSize);
  if (!DenseElementsAttr::isValidRawBuffer(shapedTypeCpp, rawBufferCpp))
    return aiirAttributeGetNull();
  return wrap(DenseElementsAttr::getFromRawBuffer(shapedTypeCpp, rawBufferCpp));
}

AiirAttribute aiirDenseElementsAttrSplatGet(AiirType shapedType,
                                            AiirAttribute element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     unwrap(element)));
}
AiirAttribute aiirDenseElementsAttrBoolSplatGet(AiirType shapedType,
                                                bool element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrUInt8SplatGet(AiirType shapedType,
                                                 uint8_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrInt8SplatGet(AiirType shapedType,
                                                int8_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrUInt32SplatGet(AiirType shapedType,
                                                  uint32_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrInt32SplatGet(AiirType shapedType,
                                                 int32_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrUInt64SplatGet(AiirType shapedType,
                                                  uint64_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrInt64SplatGet(AiirType shapedType,
                                                 int64_t element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrFloatSplatGet(AiirType shapedType,
                                                 float element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}
AiirAttribute aiirDenseElementsAttrDoubleSplatGet(AiirType shapedType,
                                                  double element) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     element));
}

AiirAttribute aiirDenseElementsAttrBoolGet(AiirType shapedType,
                                           intptr_t numElements,
                                           const int *elements) {
  SmallVector<bool, 8> values(elements, elements + numElements);
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     values));
}

/// Creates a dense attribute with elements of the type deduced by templates.
template <typename T>
static AiirAttribute getDenseAttribute(AiirType shapedType,
                                       intptr_t numElements,
                                       const T *elements) {
  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     llvm::ArrayRef(elements, numElements)));
}

AiirAttribute aiirDenseElementsAttrUInt8Get(AiirType shapedType,
                                            intptr_t numElements,
                                            const uint8_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrInt8Get(AiirType shapedType,
                                           intptr_t numElements,
                                           const int8_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrUInt16Get(AiirType shapedType,
                                             intptr_t numElements,
                                             const uint16_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrInt16Get(AiirType shapedType,
                                            intptr_t numElements,
                                            const int16_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrUInt32Get(AiirType shapedType,
                                             intptr_t numElements,
                                             const uint32_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrInt32Get(AiirType shapedType,
                                            intptr_t numElements,
                                            const int32_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrUInt64Get(AiirType shapedType,
                                             intptr_t numElements,
                                             const uint64_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrInt64Get(AiirType shapedType,
                                            intptr_t numElements,
                                            const int64_t *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrFloatGet(AiirType shapedType,
                                            intptr_t numElements,
                                            const float *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrDoubleGet(AiirType shapedType,
                                             intptr_t numElements,
                                             const double *elements) {
  return getDenseAttribute(shapedType, numElements, elements);
}
AiirAttribute aiirDenseElementsAttrBFloat16Get(AiirType shapedType,
                                               intptr_t numElements,
                                               const uint16_t *elements) {
  size_t bufferSize = numElements * 2;
  const void *buffer = static_cast<const void *>(elements);
  return aiirDenseElementsAttrRawBufferGet(shapedType, bufferSize, buffer);
}
AiirAttribute aiirDenseElementsAttrFloat16Get(AiirType shapedType,
                                              intptr_t numElements,
                                              const uint16_t *elements) {
  size_t bufferSize = numElements * 2;
  const void *buffer = static_cast<const void *>(elements);
  return aiirDenseElementsAttrRawBufferGet(shapedType, bufferSize, buffer);
}

AiirAttribute aiirDenseElementsAttrStringGet(AiirType shapedType,
                                             intptr_t numElements,
                                             AiirStringRef *strs) {
  SmallVector<StringRef, 8> values;
  values.reserve(numElements);
  for (intptr_t i = 0; i < numElements; ++i)
    values.push_back(unwrap(strs[i]));

  return wrap(DenseElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     values));
}

AiirAttribute aiirDenseElementsAttrReshapeGet(AiirAttribute attr,
                                              AiirType shapedType) {
  return wrap(llvm::cast<DenseElementsAttr>(unwrap(attr))
                  .reshape(llvm::cast<ShapedType>(unwrap(shapedType))));
}

//===----------------------------------------------------------------------===//
// Splat accessors.
//===----------------------------------------------------------------------===//

bool aiirDenseElementsAttrIsSplat(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).isSplat();
}

AiirAttribute aiirDenseElementsAttrGetSplatValue(AiirAttribute attr) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<Attribute>());
}
int aiirDenseElementsAttrGetBoolSplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<bool>();
}
int8_t aiirDenseElementsAttrGetInt8SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int8_t>();
}
uint8_t aiirDenseElementsAttrGetUInt8SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint8_t>();
}
int32_t aiirDenseElementsAttrGetInt32SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int32_t>();
}
uint32_t aiirDenseElementsAttrGetUInt32SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint32_t>();
}
int64_t aiirDenseElementsAttrGetInt64SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<int64_t>();
}
uint64_t aiirDenseElementsAttrGetUInt64SplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<uint64_t>();
}
float aiirDenseElementsAttrGetFloatSplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<float>();
}
double aiirDenseElementsAttrGetDoubleSplatValue(AiirAttribute attr) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<double>();
}
AiirStringRef aiirDenseElementsAttrGetStringSplatValue(AiirAttribute attr) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getSplatValue<StringRef>());
}

//===----------------------------------------------------------------------===//
// Indexed accessors.
//===----------------------------------------------------------------------===//

bool aiirDenseElementsAttrGetBoolValue(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<bool>()[pos];
}
int8_t aiirDenseElementsAttrGetInt8Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int8_t>()[pos];
}
uint8_t aiirDenseElementsAttrGetUInt8Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint8_t>()[pos];
}
int16_t aiirDenseElementsAttrGetInt16Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int16_t>()[pos];
}
uint16_t aiirDenseElementsAttrGetUInt16Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint16_t>()[pos];
}
int32_t aiirDenseElementsAttrGetInt32Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int32_t>()[pos];
}
uint32_t aiirDenseElementsAttrGetUInt32Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint32_t>()[pos];
}
int64_t aiirDenseElementsAttrGetInt64Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<int64_t>()[pos];
}
uint64_t aiirDenseElementsAttrGetUInt64Value(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint64_t>()[pos];
}
uint64_t aiirDenseElementsAttrGetIndexValue(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<uint64_t>()[pos];
}
float aiirDenseElementsAttrGetFloatValue(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<float>()[pos];
}
double aiirDenseElementsAttrGetDoubleValue(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<double>()[pos];
}
AiirStringRef aiirDenseElementsAttrGetStringValue(AiirAttribute attr,
                                                  intptr_t pos) {
  return wrap(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getValues<StringRef>()[pos]);
}

//===----------------------------------------------------------------------===//
// Raw data accessors.
//===----------------------------------------------------------------------===//

const void *aiirDenseElementsAttrGetRawData(AiirAttribute attr) {
  return static_cast<const void *>(
      llvm::cast<DenseElementsAttr>(unwrap(attr)).getRawData().data());
}

//===----------------------------------------------------------------------===//
// Resource blob attributes.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsADenseResourceElements(AiirAttribute attr) {
  return llvm::isa<DenseResourceElementsAttr>(unwrap(attr));
}

AiirAttribute aiirUnmanagedDenseResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, void *data, size_t dataLength,
    size_t dataAlignment, bool dataIsMutable,
    void (*deleter)(void *userData, const void *data, size_t size,
                    size_t align),
    void *userData) {
  AsmResourceBlob::DeleterFn cppDeleter = {};
  if (deleter) {
    cppDeleter = [deleter, userData](void *data, size_t size, size_t align) {
      deleter(userData, data, size, align);
    };
  }
  AsmResourceBlob blob(
      llvm::ArrayRef(static_cast<const char *>(data), dataLength),
      dataAlignment, std::move(cppDeleter), dataIsMutable);
  return wrap(
      DenseResourceElementsAttr::get(llvm::cast<ShapedType>(unwrap(shapedType)),
                                     unwrap(name), std::move(blob)));
}

AiirStringRef aiirDenseResourceElementsAttrGetName(void) {
  return wrap(DenseResourceElementsAttr::name);
}

template <typename U, typename T>
static AiirAttribute getDenseResource(AiirType shapedType, AiirStringRef name,
                                      intptr_t numElements, const T *elements) {
  return wrap(U::get(llvm::cast<ShapedType>(unwrap(shapedType)), unwrap(name),
                     UnmanagedAsmResourceBlob::allocateInferAlign(
                         llvm::ArrayRef(elements, numElements))));
}

AiirAttribute aiirUnmanagedDenseBoolResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int *elements) {
  return getDenseResource<DenseBoolResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseUInt8ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const uint8_t *elements) {
  return getDenseResource<DenseUI8ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
AiirAttribute aiirUnmanagedDenseUInt16ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const uint16_t *elements) {
  return getDenseResource<DenseUI16ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseUInt32ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const uint32_t *elements) {
  return getDenseResource<DenseUI32ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseUInt64ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const uint64_t *elements) {
  return getDenseResource<DenseUI64ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseInt8ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int8_t *elements) {
  return getDenseResource<DenseUI8ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
AiirAttribute aiirUnmanagedDenseInt16ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int16_t *elements) {
  return getDenseResource<DenseUI16ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseInt32ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int32_t *elements) {
  return getDenseResource<DenseUI32ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseInt64ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int64_t *elements) {
  return getDenseResource<DenseUI64ResourceElementsAttr>(shapedType, name,
                                                         numElements, elements);
}
AiirAttribute aiirUnmanagedDenseFloatResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const float *elements) {
  return getDenseResource<DenseF32ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
AiirAttribute aiirUnmanagedDenseDoubleResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const double *elements) {
  return getDenseResource<DenseF64ResourceElementsAttr>(shapedType, name,
                                                        numElements, elements);
}
template <typename U, typename T>
static T getDenseResourceVal(AiirAttribute attr, intptr_t pos) {
  return (*llvm::cast<U>(unwrap(attr)).tryGetAsArrayRef())[pos];
}

bool aiirDenseBoolResourceElementsAttrGetValue(AiirAttribute attr,
                                               intptr_t pos) {
  return getDenseResourceVal<DenseBoolResourceElementsAttr, uint8_t>(attr, pos);
}
uint8_t aiirDenseUInt8ResourceElementsAttrGetValue(AiirAttribute attr,
                                                   intptr_t pos) {
  return getDenseResourceVal<DenseUI8ResourceElementsAttr, uint8_t>(attr, pos);
}
uint16_t aiirDenseUInt16ResourceElementsAttrGetValue(AiirAttribute attr,
                                                     intptr_t pos) {
  return getDenseResourceVal<DenseUI16ResourceElementsAttr, uint16_t>(attr,
                                                                      pos);
}
uint32_t aiirDenseUInt32ResourceElementsAttrGetValue(AiirAttribute attr,
                                                     intptr_t pos) {
  return getDenseResourceVal<DenseUI32ResourceElementsAttr, uint32_t>(attr,
                                                                      pos);
}
uint64_t aiirDenseUInt64ResourceElementsAttrGetValue(AiirAttribute attr,
                                                     intptr_t pos) {
  return getDenseResourceVal<DenseUI64ResourceElementsAttr, uint64_t>(attr,
                                                                      pos);
}
int8_t aiirDenseInt8ResourceElementsAttrGetValue(AiirAttribute attr,
                                                 intptr_t pos) {
  return getDenseResourceVal<DenseUI8ResourceElementsAttr, int8_t>(attr, pos);
}
int16_t aiirDenseInt16ResourceElementsAttrGetValue(AiirAttribute attr,
                                                   intptr_t pos) {
  return getDenseResourceVal<DenseUI16ResourceElementsAttr, int16_t>(attr, pos);
}
int32_t aiirDenseInt32ResourceElementsAttrGetValue(AiirAttribute attr,
                                                   intptr_t pos) {
  return getDenseResourceVal<DenseUI32ResourceElementsAttr, int32_t>(attr, pos);
}
int64_t aiirDenseInt64ResourceElementsAttrGetValue(AiirAttribute attr,
                                                   intptr_t pos) {
  return getDenseResourceVal<DenseUI64ResourceElementsAttr, int64_t>(attr, pos);
}
float aiirDenseFloatResourceElementsAttrGetValue(AiirAttribute attr,
                                                 intptr_t pos) {
  return getDenseResourceVal<DenseF32ResourceElementsAttr, float>(attr, pos);
}
double aiirDenseDoubleResourceElementsAttrGetValue(AiirAttribute attr,
                                                   intptr_t pos) {
  return getDenseResourceVal<DenseF64ResourceElementsAttr, double>(attr, pos);
}

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsASparseElements(AiirAttribute attr) {
  return llvm::isa<SparseElementsAttr>(unwrap(attr));
}

AiirAttribute aiirSparseElementsAttribute(AiirType shapedType,
                                          AiirAttribute denseIndices,
                                          AiirAttribute denseValues) {
  return wrap(SparseElementsAttr::get(
      llvm::cast<ShapedType>(unwrap(shapedType)),
      llvm::cast<DenseElementsAttr>(unwrap(denseIndices)),
      llvm::cast<DenseElementsAttr>(unwrap(denseValues))));
}

AiirAttribute aiirSparseElementsAttrGetIndices(AiirAttribute attr) {
  return wrap(llvm::cast<SparseElementsAttr>(unwrap(attr)).getIndices());
}

AiirAttribute aiirSparseElementsAttrGetValues(AiirAttribute attr) {
  return wrap(llvm::cast<SparseElementsAttr>(unwrap(attr)).getValues());
}

AiirTypeID aiirSparseElementsAttrGetTypeID(void) {
  return wrap(SparseElementsAttr::getTypeID());
}

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//

bool aiirAttributeIsAStridedLayout(AiirAttribute attr) {
  return llvm::isa<StridedLayoutAttr>(unwrap(attr));
}

AiirAttribute aiirStridedLayoutAttrGet(AiirContext ctx, int64_t offset,
                                       intptr_t numStrides,
                                       const int64_t *strides) {
  return wrap(StridedLayoutAttr::get(unwrap(ctx), offset,
                                     ArrayRef<int64_t>(strides, numStrides)));
}

AiirStringRef aiirStridedLayoutAttrGetName(void) {
  return wrap(StridedLayoutAttr::name);
}

int64_t aiirStridedLayoutAttrGetOffset(AiirAttribute attr) {
  return llvm::cast<StridedLayoutAttr>(unwrap(attr)).getOffset();
}

intptr_t aiirStridedLayoutAttrGetNumStrides(AiirAttribute attr) {
  return static_cast<intptr_t>(
      llvm::cast<StridedLayoutAttr>(unwrap(attr)).getStrides().size());
}

int64_t aiirStridedLayoutAttrGetStride(AiirAttribute attr, intptr_t pos) {
  return llvm::cast<StridedLayoutAttr>(unwrap(attr)).getStrides()[pos];
}

AiirTypeID aiirStridedLayoutAttrGetTypeID(void) {
  return wrap(StridedLayoutAttr::getTypeID());
}
