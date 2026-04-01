//===-- aiir-c/BuiltinAttributes.h - C API for Builtin Attributes -*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header declares the C interface to AIIR Builtin attributes.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_BUILTINATTRIBUTES_H
#define AIIR_C_BUILTINATTRIBUTES_H

#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include "aiir-c/IntegerSet.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns an empty attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirAttributeGetNull(void);

//===----------------------------------------------------------------------===//
// Location attribute.
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool aiirAttributeIsALocation(AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Affine map attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an affine map attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAAffineMap(AiirAttribute attr);

/// Creates an affine map attribute wrapping the given map. The attribute
/// belongs to the same context as the affine map.
AIIR_CAPI_EXPORTED AiirAttribute aiirAffineMapAttrGet(AiirAffineMap map);

AIIR_CAPI_EXPORTED AiirStringRef aiirAffineMapAttrGetName(void);

/// Returns the affine map wrapped in the given affine map attribute.
AIIR_CAPI_EXPORTED AiirAffineMap aiirAffineMapAttrGetValue(AiirAttribute attr);

/// Returns the typeID of an AffineMap attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirAffineMapAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Array attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an array attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAArray(AiirAttribute attr);

/// Creates an array element containing the given list of elements in the given
/// context.
AIIR_CAPI_EXPORTED AiirAttribute aiirArrayAttrGet(
    AiirContext ctx, intptr_t numElements, AiirAttribute const *elements);

AIIR_CAPI_EXPORTED AiirStringRef aiirArrayAttrGetName(void);

/// Returns the number of elements stored in the given array attribute.
AIIR_CAPI_EXPORTED intptr_t aiirArrayAttrGetNumElements(AiirAttribute attr);

/// Returns pos-th element stored in the given array attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirArrayAttrGetElement(AiirAttribute attr,
                                                         intptr_t pos);

/// Returns the typeID of an Array attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirArrayAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Dictionary attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a dictionary attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsADictionary(AiirAttribute attr);

/// Creates a dictionary attribute containing the given list of elements in the
/// provided context.
AIIR_CAPI_EXPORTED AiirAttribute aiirDictionaryAttrGet(
    AiirContext ctx, intptr_t numElements, AiirNamedAttribute const *elements);

AIIR_CAPI_EXPORTED AiirStringRef aiirDictionaryAttrGetName(void);

/// Returns the number of attributes contained in a dictionary attribute.
AIIR_CAPI_EXPORTED intptr_t
aiirDictionaryAttrGetNumElements(AiirAttribute attr);

/// Returns pos-th element of the given dictionary attribute.
AIIR_CAPI_EXPORTED AiirNamedAttribute
aiirDictionaryAttrGetElement(AiirAttribute attr, intptr_t pos);

/// Returns the dictionary attribute element with the given name or NULL if the
/// given name does not exist in the dictionary.
AIIR_CAPI_EXPORTED AiirAttribute
aiirDictionaryAttrGetElementByName(AiirAttribute attr, AiirStringRef name);

/// Returns the typeID of a Dictionary attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirDictionaryAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Floating point attribute.
//===----------------------------------------------------------------------===//

// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is a floating point attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAFloat(AiirAttribute attr);

AIIR_CAPI_EXPORTED AiirStringRef aiirFloatAttrGetName(void);

/// Creates a floating point attribute in the given context with the given
/// double value and double-precision FP semantics.
AIIR_CAPI_EXPORTED AiirAttribute aiirFloatAttrDoubleGet(AiirContext ctx,
                                                        AiirType type,
                                                        double value);

/// Same as "aiirFloatAttrDoubleGet", but if the type is not valid for a
/// construction of a FloatAttr, returns a null AiirAttribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirFloatAttrDoubleGetChecked(AiirLocation loc,
                                                               AiirType type,
                                                               double value);

/// Returns the value stored in the given floating point attribute, interpreting
/// the value as double.
AIIR_CAPI_EXPORTED double aiirFloatAttrGetValueDouble(AiirAttribute attr);

/// Returns the typeID of a Float attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirFloatAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Integer attribute.
//===----------------------------------------------------------------------===//

// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is an integer attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAInteger(AiirAttribute attr);

/// Creates an integer attribute of the given type with the given integer
/// value.
AIIR_CAPI_EXPORTED AiirAttribute aiirIntegerAttrGet(AiirType type,
                                                    int64_t value);

AIIR_CAPI_EXPORTED AiirStringRef aiirIntegerAttrGetName(void);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of signless type and fits into a signed 64-bit integer.
AIIR_CAPI_EXPORTED int64_t aiirIntegerAttrGetValueInt(AiirAttribute attr);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of signed type and fits into a signed 64-bit integer.
AIIR_CAPI_EXPORTED int64_t aiirIntegerAttrGetValueSInt(AiirAttribute attr);

/// Returns the value stored in the given integer attribute, assuming the value
/// is of unsigned type and fits into an unsigned 64-bit integer.
AIIR_CAPI_EXPORTED uint64_t aiirIntegerAttrGetValueUInt(AiirAttribute attr);

/// Returns the bit width of the integer attribute's underlying APInt value.
/// This is useful for determining the size of the integer, especially for
/// values larger than 64 bits.
AIIR_CAPI_EXPORTED unsigned aiirIntegerAttrGetValueBitWidth(AiirAttribute attr);

/// Returns the number of 64-bit words that make up the integer attribute's
/// underlying APInt value. For integers <= 64 bits, this returns 1.
AIIR_CAPI_EXPORTED unsigned aiirIntegerAttrGetValueNumWords(AiirAttribute attr);

/// Copies the 64-bit words making up the integer attribute's APInt value into
/// the provided buffer. The buffer must have space for at least
/// aiirIntegerAttrGetValueNumWords(attr) elements. Words are stored in
/// little-endian order (least significant word first). The sign information
/// is not encoded in the words themselves; use the type's signedness to
/// interpret the value correctly.
AIIR_CAPI_EXPORTED void aiirIntegerAttrGetValueWords(AiirAttribute attr,
                                                     uint64_t *words);

/// Creates an integer attribute of the given type from an array of 64-bit
/// words. This is useful for creating integer attributes with values with
/// widths larger than 64 bits. Words are in little-endian order (least
/// significant word first). The number of words must match the bit width of the
/// type: numWords = ceil(bitWidth / 64).
AIIR_CAPI_EXPORTED AiirAttribute aiirIntegerAttrGetFromWords(
    AiirType type, unsigned numWords, const uint64_t *words);

/// Returns the typeID of an Integer attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirIntegerAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Bool attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a bool attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsABool(AiirAttribute attr);

/// Creates a bool attribute in the given context with the given value.
AIIR_CAPI_EXPORTED AiirAttribute aiirBoolAttrGet(AiirContext ctx, int value);

/// Returns the value stored in the given bool attribute.
AIIR_CAPI_EXPORTED bool aiirBoolAttrGetValue(AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Integer set attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an integer set attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAIntegerSet(AiirAttribute attr);

/// Creates an integer set attribute wrapping the given set. The attribute
/// belongs to the same context as the integer set.
AIIR_CAPI_EXPORTED AiirAttribute aiirIntegerSetAttrGet(AiirIntegerSet set);

AIIR_CAPI_EXPORTED AiirStringRef aiirIntegerSetAttrGetName(void);

/// Returns the integer set wrapped in the given integer set attribute.
AIIR_CAPI_EXPORTED AiirIntegerSet
aiirIntegerSetAttrGetValue(AiirAttribute attr);

/// Returns the typeID of an IntegerSet attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirIntegerSetAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Opaque attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an opaque attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAOpaque(AiirAttribute attr);

/// Creates an opaque attribute in the given context associated with the dialect
/// identified by its namespace. The attribute contains opaque byte data of the
/// specified length (data need not be null-terminated).
AIIR_CAPI_EXPORTED AiirAttribute
aiirOpaqueAttrGet(AiirContext ctx, AiirStringRef dialectNamespace,
                  intptr_t dataLength, const char *data, AiirType type);

AIIR_CAPI_EXPORTED AiirStringRef aiirOpaqueAttrGetName(void);

/// Returns the namespace of the dialect with which the given opaque attribute
/// is associated. The namespace string is owned by the context.
AIIR_CAPI_EXPORTED AiirStringRef
aiirOpaqueAttrGetDialectNamespace(AiirAttribute attr);

/// Returns the raw data as a string reference. The data remains live as long as
/// the context in which the attribute lives.
AIIR_CAPI_EXPORTED AiirStringRef aiirOpaqueAttrGetData(AiirAttribute attr);

/// Returns the typeID of an Opaque attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirOpaqueAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// String attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a string attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAString(AiirAttribute attr);

/// Creates a string attribute in the given context containing the given string.

AIIR_CAPI_EXPORTED AiirAttribute aiirStringAttrGet(AiirContext ctx,
                                                   AiirStringRef str);

AIIR_CAPI_EXPORTED AiirStringRef aiirStringAttrGetName(void);

/// Creates a string attribute in the given context containing the given string.
/// Additionally, the attribute has the given type.
AIIR_CAPI_EXPORTED AiirAttribute aiirStringAttrTypedGet(AiirType type,
                                                        AiirStringRef str);

/// Returns the attribute values as a string reference. The data remains live as
/// long as the context in which the attribute lives.
AIIR_CAPI_EXPORTED AiirStringRef aiirStringAttrGetValue(AiirAttribute attr);

/// Returns the typeID of a String attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirStringAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// SymbolRef attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a symbol reference attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsASymbolRef(AiirAttribute attr);

/// Creates a symbol reference attribute in the given context referencing a
/// symbol identified by the given string inside a list of nested references.
/// Each of the references in the list must not be nested.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSymbolRefAttrGet(AiirContext ctx, AiirStringRef symbol,
                     intptr_t numReferences, AiirAttribute const *references);

AIIR_CAPI_EXPORTED AiirStringRef aiirSymbolRefAttrGetName(void);

/// Returns the string reference to the root referenced symbol. The data remains
/// live as long as the context in which the attribute lives.
AIIR_CAPI_EXPORTED AiirStringRef
aiirSymbolRefAttrGetRootReference(AiirAttribute attr);

/// Returns the string reference to the leaf referenced symbol. The data remains
/// live as long as the context in which the attribute lives.
AIIR_CAPI_EXPORTED AiirStringRef
aiirSymbolRefAttrGetLeafReference(AiirAttribute attr);

/// Returns the number of references nested in the given symbol reference
/// attribute.
AIIR_CAPI_EXPORTED intptr_t
aiirSymbolRefAttrGetNumNestedReferences(AiirAttribute attr);

/// Returns pos-th reference nested in the given symbol reference attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSymbolRefAttrGetNestedReference(AiirAttribute attr, intptr_t pos);

/// Returns the typeID of an SymbolRef attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirSymbolRefAttrGetTypeID(void);

/// Creates a DistinctAttr with the referenced attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirDistinctAttrCreate(AiirAttribute referencedAttr);

//===----------------------------------------------------------------------===//
// Flat SymbolRef attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a flat symbol reference attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAFlatSymbolRef(AiirAttribute attr);

/// Creates a flat symbol reference attribute in the given context referencing a
/// symbol identified by the given string.
AIIR_CAPI_EXPORTED AiirAttribute aiirFlatSymbolRefAttrGet(AiirContext ctx,
                                                          AiirStringRef symbol);

AIIR_CAPI_EXPORTED AiirStringRef aiirFlatSymbolRefAttrGetName(void);

/// Returns the referenced symbol as a string reference. The data remains live
/// as long as the context in which the attribute lives.
AIIR_CAPI_EXPORTED AiirStringRef
aiirFlatSymbolRefAttrGetValue(AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Type attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a type attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAType(AiirAttribute attr);

/// Creates a type attribute wrapping the given type in the same context as the
/// type.
AIIR_CAPI_EXPORTED AiirAttribute aiirTypeAttrGet(AiirType type);

AIIR_CAPI_EXPORTED AiirStringRef aiirTypeAttrGetName(void);

/// Returns the type stored in the given type attribute.
AIIR_CAPI_EXPORTED AiirType aiirTypeAttrGetValue(AiirAttribute attr);

/// Returns the typeID of a Type attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirTypeAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Unit attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a unit attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAUnit(AiirAttribute attr);

/// Creates a unit attribute in the given context.
AIIR_CAPI_EXPORTED AiirAttribute aiirUnitAttrGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirUnitAttrGetName(void);

/// Returns the typeID of a Unit attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirUnitAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Elements attributes.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is an elements attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAElements(AiirAttribute attr);

/// Returns the element at the given rank-dimensional index.
AIIR_CAPI_EXPORTED AiirAttribute aiirElementsAttrGetValue(AiirAttribute attr,
                                                          intptr_t rank,
                                                          uint64_t *idxs);

/// Checks whether the given rank-dimensional index is valid in the given
/// elements attribute.
AIIR_CAPI_EXPORTED bool
aiirElementsAttrIsValidIndex(AiirAttribute attr, intptr_t rank, uint64_t *idxs);

/// Gets the total number of elements in the given elements attribute. In order
/// to iterate over the attribute, obtain its type, which must be a statically
/// shaped type and use its sizes to build a multi-dimensional index.
AIIR_CAPI_EXPORTED int64_t aiirElementsAttrGetNumElements(AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Dense array attribute.
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED AiirTypeID aiirDenseArrayAttrGetTypeID(void);

/// Checks whether the given attribute is a dense array attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseBoolArray(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseI8Array(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseI16Array(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseI32Array(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseI64Array(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseF32Array(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseF64Array(AiirAttribute attr);

/// Create a dense array attribute with the given elements.
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseBoolArrayGet(AiirContext ctx,
                                                       intptr_t size,
                                                       int const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseI8ArrayGet(AiirContext ctx,
                                                     intptr_t size,
                                                     int8_t const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseI16ArrayGet(AiirContext ctx,
                                                      intptr_t size,
                                                      int16_t const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseI32ArrayGet(AiirContext ctx,
                                                      intptr_t size,
                                                      int32_t const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseI64ArrayGet(AiirContext ctx,
                                                      intptr_t size,
                                                      int64_t const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseF32ArrayGet(AiirContext ctx,
                                                      intptr_t size,
                                                      float const *values);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseF64ArrayGet(AiirContext ctx,
                                                      intptr_t size,
                                                      double const *values);

/// Get the size of a dense array.
AIIR_CAPI_EXPORTED intptr_t aiirDenseArrayGetNumElements(AiirAttribute attr);

/// Get an element of a dense array.
AIIR_CAPI_EXPORTED bool aiirDenseBoolArrayGetElement(AiirAttribute attr,
                                                     intptr_t pos);
AIIR_CAPI_EXPORTED int8_t aiirDenseI8ArrayGetElement(AiirAttribute attr,
                                                     intptr_t pos);
AIIR_CAPI_EXPORTED int16_t aiirDenseI16ArrayGetElement(AiirAttribute attr,
                                                       intptr_t pos);
AIIR_CAPI_EXPORTED int32_t aiirDenseI32ArrayGetElement(AiirAttribute attr,
                                                       intptr_t pos);
AIIR_CAPI_EXPORTED int64_t aiirDenseI64ArrayGetElement(AiirAttribute attr,
                                                       intptr_t pos);
AIIR_CAPI_EXPORTED float aiirDenseF32ArrayGetElement(AiirAttribute attr,
                                                     intptr_t pos);
AIIR_CAPI_EXPORTED double aiirDenseF64ArrayGetElement(AiirAttribute attr,
                                                      intptr_t pos);

//===----------------------------------------------------------------------===//
// Dense elements attribute.
//===----------------------------------------------------------------------===//

// TODO: decide on the interface and add support for complex elements.
// TODO: add support for APFloat and APInt to LLVM IR C API, then expose the
// relevant functions here.

/// Checks whether the given attribute is a dense elements attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseElements(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseIntElements(AiirAttribute attr);
AIIR_CAPI_EXPORTED bool aiirAttributeIsADenseFPElements(AiirAttribute attr);

/// Returns the typeID of a DenseTypedElements attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirDenseTypedElementsAttrGetTypeID(void);
/// Deprecated API. Will be removed in the future.
AIIR_CAPI_EXPORTED AiirTypeID aiirDenseIntOrFPElementsAttrGetTypeID(void);

/// Creates a dense elements attribute with the given Shaped type and elements
/// in the same context as the type.
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrGet(
    AiirType shapedType, intptr_t numElements, AiirAttribute const *elements);

/// Creates a dense elements attribute with the given Shaped type and elements
/// populated from a packed, row-major opaque buffer of contents.
///
/// The format of the raw buffer is a densely packed array of values that
/// can be bitcast to the storage format of the element type specified.
/// Types that are not byte aligned will be:
///   - For bitwidth > 1: Rounded up to the next byte.
///   - For bitwidth = 1: Packed into 8bit bytes with bits corresponding to
///     the linear order of the shape type from MSB to LSB, padded to on the
///     right.
///
/// A raw buffer of a single element (or for 1-bit, a byte of value 0 or 255)
/// will be interpreted as a splat. User code should be prepared for additional,
/// conformant patterns to be identified as splats in the future.
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrRawBufferGet(
    AiirType shapedType, size_t rawBufferSize, const void *rawBuffer);

/// Creates a dense elements attribute with the given Shaped type containing a
/// single replicated element (splat).
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrSplatGet(AiirType shapedType, AiirAttribute element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrBoolSplatGet(AiirType shapedType, bool element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrUInt8SplatGet(AiirType shapedType, uint8_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrInt8SplatGet(AiirType shapedType, int8_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrUInt32SplatGet(AiirType shapedType, uint32_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrInt32SplatGet(AiirType shapedType, int32_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrUInt64SplatGet(AiirType shapedType, uint64_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrInt64SplatGet(AiirType shapedType, int64_t element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrFloatSplatGet(AiirType shapedType, float element);
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrDoubleSplatGet(AiirType shapedType, double element);

/// Creates a dense elements attribute with the given shaped type from elements
/// of a specific type. Expects the element type of the shaped type to match the
/// data element type.
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrBoolGet(
    AiirType shapedType, intptr_t numElements, const int *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrUInt8Get(
    AiirType shapedType, intptr_t numElements, const uint8_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrInt8Get(
    AiirType shapedType, intptr_t numElements, const int8_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrUInt16Get(
    AiirType shapedType, intptr_t numElements, const uint16_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrInt16Get(
    AiirType shapedType, intptr_t numElements, const int16_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrUInt32Get(
    AiirType shapedType, intptr_t numElements, const uint32_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrInt32Get(
    AiirType shapedType, intptr_t numElements, const int32_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrUInt64Get(
    AiirType shapedType, intptr_t numElements, const uint64_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrInt64Get(
    AiirType shapedType, intptr_t numElements, const int64_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrFloatGet(
    AiirType shapedType, intptr_t numElements, const float *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrDoubleGet(
    AiirType shapedType, intptr_t numElements, const double *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrBFloat16Get(
    AiirType shapedType, intptr_t numElements, const uint16_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrFloat16Get(
    AiirType shapedType, intptr_t numElements, const uint16_t *elements);

/// Creates a dense elements attribute with the given shaped type from string
/// elements.
AIIR_CAPI_EXPORTED AiirAttribute aiirDenseElementsAttrStringGet(
    AiirType shapedType, intptr_t numElements, AiirStringRef *strs);

/// Creates a dense elements attribute that has the same data as the given dense
/// elements attribute and a different shaped type. The new type must have the
/// same total number of elements.
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrReshapeGet(AiirAttribute attr, AiirType shapedType);

/// Checks whether the given dense elements attribute contains a single
/// replicated value (splat).
AIIR_CAPI_EXPORTED bool aiirDenseElementsAttrIsSplat(AiirAttribute attr);

/// Returns the single replicated value (splat) of a specific type contained by
/// the given dense elements attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirDenseElementsAttrGetSplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED int
aiirDenseElementsAttrGetBoolSplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED int8_t
aiirDenseElementsAttrGetInt8SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED uint8_t
aiirDenseElementsAttrGetUInt8SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED int32_t
aiirDenseElementsAttrGetInt32SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED uint32_t
aiirDenseElementsAttrGetUInt32SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED int64_t
aiirDenseElementsAttrGetInt64SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED uint64_t
aiirDenseElementsAttrGetUInt64SplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED float
aiirDenseElementsAttrGetFloatSplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED double
aiirDenseElementsAttrGetDoubleSplatValue(AiirAttribute attr);
AIIR_CAPI_EXPORTED AiirStringRef
aiirDenseElementsAttrGetStringSplatValue(AiirAttribute attr);

/// Returns the pos-th value (flat contiguous indexing) of a specific type
/// contained by the given dense elements attribute.
AIIR_CAPI_EXPORTED bool aiirDenseElementsAttrGetBoolValue(AiirAttribute attr,
                                                          intptr_t pos);
AIIR_CAPI_EXPORTED int8_t aiirDenseElementsAttrGetInt8Value(AiirAttribute attr,
                                                            intptr_t pos);
AIIR_CAPI_EXPORTED uint8_t
aiirDenseElementsAttrGetUInt8Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int16_t
aiirDenseElementsAttrGetInt16Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint16_t
aiirDenseElementsAttrGetUInt16Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int32_t
aiirDenseElementsAttrGetInt32Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint32_t
aiirDenseElementsAttrGetUInt32Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int64_t
aiirDenseElementsAttrGetInt64Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint64_t
aiirDenseElementsAttrGetUInt64Value(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint64_t
aiirDenseElementsAttrGetIndexValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED float aiirDenseElementsAttrGetFloatValue(AiirAttribute attr,
                                                            intptr_t pos);
AIIR_CAPI_EXPORTED double
aiirDenseElementsAttrGetDoubleValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED AiirStringRef
aiirDenseElementsAttrGetStringValue(AiirAttribute attr, intptr_t pos);

/// Returns the raw data of the given dense elements attribute.
AIIR_CAPI_EXPORTED const void *
aiirDenseElementsAttrGetRawData(AiirAttribute attr);

//===----------------------------------------------------------------------===//
// Resource blob attributes.
//===----------------------------------------------------------------------===//

AIIR_CAPI_EXPORTED bool
aiirAttributeIsADenseResourceElements(AiirAttribute attr);

/// Unlike the typed accessors below, constructs the attribute with a raw
/// data buffer and no type/alignment checking. Use a more strongly typed
/// accessor if possible. If dataIsMutable is false, then an immutable
/// AsmResourceBlob will be created and that passed data contents will be
/// treated as const.
/// If the deleter is non NULL, then it will be called when the data buffer
/// can no longer be accessed (passing userData to it).
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, void *data, size_t dataLength,
    size_t dataAlignment, bool dataIsMutable,
    void (*deleter)(void *userData, const void *data, size_t size,
                    size_t align),
    void *userData);

AIIR_CAPI_EXPORTED AiirStringRef aiirDenseResourceElementsAttrGetName(void);

AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseBoolResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseUInt8ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const uint8_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseInt8ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int8_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute
aiirUnmanagedDenseUInt16ResourceElementsAttrGet(AiirType shapedType,
                                                AiirStringRef name,
                                                intptr_t numElements,
                                                const uint16_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseInt16ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int16_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute
aiirUnmanagedDenseUInt32ResourceElementsAttrGet(AiirType shapedType,
                                                AiirStringRef name,
                                                intptr_t numElements,
                                                const uint32_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseInt32ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int32_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute
aiirUnmanagedDenseUInt64ResourceElementsAttrGet(AiirType shapedType,
                                                AiirStringRef name,
                                                intptr_t numElements,
                                                const uint64_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseInt64ResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const int64_t *elements);
AIIR_CAPI_EXPORTED AiirAttribute aiirUnmanagedDenseFloatResourceElementsAttrGet(
    AiirType shapedType, AiirStringRef name, intptr_t numElements,
    const float *elements);
AIIR_CAPI_EXPORTED AiirAttribute
aiirUnmanagedDenseDoubleResourceElementsAttrGet(AiirType shapedType,
                                                AiirStringRef name,
                                                intptr_t numElements,
                                                const double *elements);

/// Returns the pos-th value (flat contiguous indexing) of a specific type
/// contained by the given dense resource elements attribute.
AIIR_CAPI_EXPORTED bool
aiirDenseBoolResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int8_t
aiirDenseInt8ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint8_t
aiirDenseUInt8ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int16_t
aiirDenseInt16ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint16_t
aiirDenseUInt16ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int32_t
aiirDenseInt32ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint32_t
aiirDenseUInt32ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED int64_t
aiirDenseInt64ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED uint64_t
aiirDenseUInt64ResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED float
aiirDenseFloatResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);
AIIR_CAPI_EXPORTED double
aiirDenseDoubleResourceElementsAttrGetValue(AiirAttribute attr, intptr_t pos);

//===----------------------------------------------------------------------===//
// Sparse elements attribute.
//===----------------------------------------------------------------------===//

/// Checks whether the given attribute is a sparse elements attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsASparseElements(AiirAttribute attr);

/// Creates a sparse elements attribute of the given shape from a list of
/// indices and a list of associated values. Both lists are expected to be dense
/// elements attributes with the same number of elements. The list of indices is
/// expected to contain 64-bit integers. The attribute is created in the same
/// context as the type.
AIIR_CAPI_EXPORTED AiirAttribute aiirSparseElementsAttribute(
    AiirType shapedType, AiirAttribute denseIndices, AiirAttribute denseValues);

/// Returns the dense elements attribute containing 64-bit integer indices of
/// non-null elements in the given sparse elements attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSparseElementsAttrGetIndices(AiirAttribute attr);

/// Returns the dense elements attribute containing the non-null elements in the
/// given sparse elements attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirSparseElementsAttrGetValues(AiirAttribute attr);

/// Returns the typeID of a SparseElements attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirSparseElementsAttrGetTypeID(void);

//===----------------------------------------------------------------------===//
// Strided layout attribute.
//===----------------------------------------------------------------------===//

// Checks wheather the given attribute is a strided layout attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsAStridedLayout(AiirAttribute attr);

// Creates a strided layout attribute from given strides and offset.
AIIR_CAPI_EXPORTED AiirAttribute
aiirStridedLayoutAttrGet(AiirContext ctx, int64_t offset, intptr_t numStrides,
                         const int64_t *strides);

AIIR_CAPI_EXPORTED AiirStringRef aiirStridedLayoutAttrGetName(void);

// Returns the offset in the given strided layout layout attribute.
AIIR_CAPI_EXPORTED int64_t aiirStridedLayoutAttrGetOffset(AiirAttribute attr);

// Returns the number of strides in the given strided layout attribute.
AIIR_CAPI_EXPORTED intptr_t
aiirStridedLayoutAttrGetNumStrides(AiirAttribute attr);

// Returns the pos-th stride stored in the given strided layout attribute.
AIIR_CAPI_EXPORTED int64_t aiirStridedLayoutAttrGetStride(AiirAttribute attr,
                                                          intptr_t pos);

/// Returns the typeID of a StridedLayout attribute.
AIIR_CAPI_EXPORTED AiirTypeID aiirStridedLayoutAttrGetTypeID(void);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_BUILTINATTRIBUTES_H
