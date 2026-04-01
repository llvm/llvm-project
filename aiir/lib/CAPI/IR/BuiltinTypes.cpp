//===- BuiltinTypes.cpp - C Interface to AIIR Builtin Types ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/BuiltinTypes.h"
#include "aiir-c/AffineMap.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/AffineMap.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/IR/AffineMap.h"
#include "aiir/IR/BuiltinTypes.h"
#include "aiir/IR/Types.h"

#include <algorithm>

using namespace aiir;

//===----------------------------------------------------------------------===//
// Integer types.
//===----------------------------------------------------------------------===//

AiirTypeID aiirIntegerTypeGetTypeID() { return wrap(IntegerType::getTypeID()); }

bool aiirTypeIsAInteger(AiirType type) {
  return llvm::isa<IntegerType>(unwrap(type));
}

AiirType aiirIntegerTypeGet(AiirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth));
}

AiirStringRef aiirIntegerTypeGetName(void) { return wrap(IntegerType::name); }

AiirType aiirIntegerTypeSignedGet(AiirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth, IntegerType::Signed));
}

AiirType aiirIntegerTypeUnsignedGet(AiirContext ctx, unsigned bitwidth) {
  return wrap(IntegerType::get(unwrap(ctx), bitwidth, IntegerType::Unsigned));
}

unsigned aiirIntegerTypeGetWidth(AiirType type) {
  return llvm::cast<IntegerType>(unwrap(type)).getWidth();
}

bool aiirIntegerTypeIsSignless(AiirType type) {
  return llvm::cast<IntegerType>(unwrap(type)).isSignless();
}

bool aiirIntegerTypeIsSigned(AiirType type) {
  return llvm::cast<IntegerType>(unwrap(type)).isSigned();
}

bool aiirIntegerTypeIsUnsigned(AiirType type) {
  return llvm::cast<IntegerType>(unwrap(type)).isUnsigned();
}

//===----------------------------------------------------------------------===//
// Index type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirIndexTypeGetTypeID() { return wrap(IndexType::getTypeID()); }

bool aiirTypeIsAIndex(AiirType type) {
  return llvm::isa<IndexType>(unwrap(type));
}

AiirType aiirIndexTypeGet(AiirContext ctx) {
  return wrap(IndexType::get(unwrap(ctx)));
}

AiirStringRef aiirIndexTypeGetName(void) { return wrap(IndexType::name); }

//===----------------------------------------------------------------------===//
// Floating-point types.
//===----------------------------------------------------------------------===//

bool aiirTypeIsAFloat(AiirType type) {
  return llvm::isa<FloatType>(unwrap(type));
}

unsigned aiirFloatTypeGetWidth(AiirType type) {
  return llvm::cast<FloatType>(unwrap(type)).getWidth();
}

AiirTypeID aiirFloat4E2M1FNTypeGetTypeID() {
  return wrap(Float4E2M1FNType::getTypeID());
}

bool aiirTypeIsAFloat4E2M1FN(AiirType type) {
  return llvm::isa<Float4E2M1FNType>(unwrap(type));
}

AiirType aiirFloat4E2M1FNTypeGet(AiirContext ctx) {
  return wrap(Float4E2M1FNType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat4E2M1FNTypeGetName(void) {
  return wrap(Float4E2M1FNType::name);
}

AiirTypeID aiirFloat6E2M3FNTypeGetTypeID() {
  return wrap(Float6E2M3FNType::getTypeID());
}

bool aiirTypeIsAFloat6E2M3FN(AiirType type) {
  return llvm::isa<Float6E2M3FNType>(unwrap(type));
}

AiirType aiirFloat6E2M3FNTypeGet(AiirContext ctx) {
  return wrap(Float6E2M3FNType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat6E2M3FNTypeGetName(void) {
  return wrap(Float6E2M3FNType::name);
}

AiirTypeID aiirFloat6E3M2FNTypeGetTypeID() {
  return wrap(Float6E3M2FNType::getTypeID());
}

bool aiirTypeIsAFloat6E3M2FN(AiirType type) {
  return llvm::isa<Float6E3M2FNType>(unwrap(type));
}

AiirType aiirFloat6E3M2FNTypeGet(AiirContext ctx) {
  return wrap(Float6E3M2FNType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat6E3M2FNTypeGetName(void) {
  return wrap(Float6E3M2FNType::name);
}

AiirTypeID aiirFloat8E5M2TypeGetTypeID() {
  return wrap(Float8E5M2Type::getTypeID());
}

bool aiirTypeIsAFloat8E5M2(AiirType type) {
  return llvm::isa<Float8E5M2Type>(unwrap(type));
}

AiirType aiirFloat8E5M2TypeGet(AiirContext ctx) {
  return wrap(Float8E5M2Type::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E5M2TypeGetName(void) {
  return wrap(Float8E5M2Type::name);
}

AiirTypeID aiirFloat8E4M3TypeGetTypeID() {
  return wrap(Float8E4M3Type::getTypeID());
}

bool aiirTypeIsAFloat8E4M3(AiirType type) {
  return llvm::isa<Float8E4M3Type>(unwrap(type));
}

AiirType aiirFloat8E4M3TypeGet(AiirContext ctx) {
  return wrap(Float8E4M3Type::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E4M3TypeGetName(void) {
  return wrap(Float8E4M3Type::name);
}

AiirTypeID aiirFloat8E4M3FNTypeGetTypeID() {
  return wrap(Float8E4M3FNType::getTypeID());
}

bool aiirTypeIsAFloat8E4M3FN(AiirType type) {
  return llvm::isa<Float8E4M3FNType>(unwrap(type));
}

AiirType aiirFloat8E4M3FNTypeGet(AiirContext ctx) {
  return wrap(Float8E4M3FNType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E4M3FNTypeGetName(void) {
  return wrap(Float8E4M3FNType::name);
}

AiirTypeID aiirFloat8E5M2FNUZTypeGetTypeID() {
  return wrap(Float8E5M2FNUZType::getTypeID());
}

bool aiirTypeIsAFloat8E5M2FNUZ(AiirType type) {
  return llvm::isa<Float8E5M2FNUZType>(unwrap(type));
}

AiirType aiirFloat8E5M2FNUZTypeGet(AiirContext ctx) {
  return wrap(Float8E5M2FNUZType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E5M2FNUZTypeGetName(void) {
  return wrap(Float8E5M2FNUZType::name);
}

AiirTypeID aiirFloat8E4M3FNUZTypeGetTypeID() {
  return wrap(Float8E4M3FNUZType::getTypeID());
}

bool aiirTypeIsAFloat8E4M3FNUZ(AiirType type) {
  return llvm::isa<Float8E4M3FNUZType>(unwrap(type));
}

AiirType aiirFloat8E4M3FNUZTypeGet(AiirContext ctx) {
  return wrap(Float8E4M3FNUZType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E4M3FNUZTypeGetName(void) {
  return wrap(Float8E4M3FNUZType::name);
}

AiirTypeID aiirFloat8E4M3B11FNUZTypeGetTypeID() {
  return wrap(Float8E4M3B11FNUZType::getTypeID());
}

bool aiirTypeIsAFloat8E4M3B11FNUZ(AiirType type) {
  return llvm::isa<Float8E4M3B11FNUZType>(unwrap(type));
}

AiirType aiirFloat8E4M3B11FNUZTypeGet(AiirContext ctx) {
  return wrap(Float8E4M3B11FNUZType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E4M3B11FNUZTypeGetName(void) {
  return wrap(Float8E4M3B11FNUZType::name);
}

AiirTypeID aiirFloat8E3M4TypeGetTypeID() {
  return wrap(Float8E3M4Type::getTypeID());
}

bool aiirTypeIsAFloat8E3M4(AiirType type) {
  return llvm::isa<Float8E3M4Type>(unwrap(type));
}

AiirType aiirFloat8E3M4TypeGet(AiirContext ctx) {
  return wrap(Float8E3M4Type::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E3M4TypeGetName(void) {
  return wrap(Float8E3M4Type::name);
}

AiirTypeID aiirFloat8E8M0FNUTypeGetTypeID() {
  return wrap(Float8E8M0FNUType::getTypeID());
}

bool aiirTypeIsAFloat8E8M0FNU(AiirType type) {
  return llvm::isa<Float8E8M0FNUType>(unwrap(type));
}

AiirType aiirFloat8E8M0FNUTypeGet(AiirContext ctx) {
  return wrap(Float8E8M0FNUType::get(unwrap(ctx)));
}

AiirStringRef aiirFloat8E8M0FNUTypeGetName(void) {
  return wrap(Float8E8M0FNUType::name);
}

AiirTypeID aiirBFloat16TypeGetTypeID() {
  return wrap(BFloat16Type::getTypeID());
}

bool aiirTypeIsABF16(AiirType type) {
  return llvm::isa<BFloat16Type>(unwrap(type));
}

AiirType aiirBF16TypeGet(AiirContext ctx) {
  return wrap(BFloat16Type::get(unwrap(ctx)));
}

AiirStringRef aiirBF16TypeGetName(void) { return wrap(BFloat16Type::name); }

AiirTypeID aiirFloat16TypeGetTypeID() { return wrap(Float16Type::getTypeID()); }

bool aiirTypeIsAF16(AiirType type) {
  return llvm::isa<Float16Type>(unwrap(type));
}

AiirType aiirF16TypeGet(AiirContext ctx) {
  return wrap(Float16Type::get(unwrap(ctx)));
}

AiirStringRef aiirF16TypeGetName(void) { return wrap(Float16Type::name); }

AiirTypeID aiirFloatTF32TypeGetTypeID() {
  return wrap(FloatTF32Type::getTypeID());
}

bool aiirTypeIsATF32(AiirType type) {
  return llvm::isa<FloatTF32Type>(unwrap(type));
}

AiirType aiirTF32TypeGet(AiirContext ctx) {
  return wrap(FloatTF32Type::get(unwrap(ctx)));
}

AiirStringRef aiirTF32TypeGetName(void) { return wrap(FloatTF32Type::name); }

AiirTypeID aiirFloat32TypeGetTypeID() { return wrap(Float32Type::getTypeID()); }

bool aiirTypeIsAF32(AiirType type) {
  return llvm::isa<Float32Type>(unwrap(type));
}

AiirType aiirF32TypeGet(AiirContext ctx) {
  return wrap(Float32Type::get(unwrap(ctx)));
}

AiirStringRef aiirF32TypeGetName(void) { return wrap(Float32Type::name); }

AiirTypeID aiirFloat64TypeGetTypeID() { return wrap(Float64Type::getTypeID()); }

bool aiirTypeIsAF64(AiirType type) {
  return llvm::isa<Float64Type>(unwrap(type));
}

AiirType aiirF64TypeGet(AiirContext ctx) {
  return wrap(Float64Type::get(unwrap(ctx)));
}

AiirStringRef aiirF64TypeGetName(void) { return wrap(Float64Type::name); }

//===----------------------------------------------------------------------===//
// None type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirNoneTypeGetTypeID() { return wrap(NoneType::getTypeID()); }

bool aiirTypeIsANone(AiirType type) {
  return llvm::isa<NoneType>(unwrap(type));
}

AiirType aiirNoneTypeGet(AiirContext ctx) {
  return wrap(NoneType::get(unwrap(ctx)));
}

AiirStringRef aiirNoneTypeGetName(void) { return wrap(NoneType::name); }

//===----------------------------------------------------------------------===//
// Complex type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirComplexTypeGetTypeID() { return wrap(ComplexType::getTypeID()); }

bool aiirTypeIsAComplex(AiirType type) {
  return llvm::isa<ComplexType>(unwrap(type));
}

AiirType aiirComplexTypeGet(AiirType elementType) {
  return wrap(ComplexType::get(unwrap(elementType)));
}

AiirStringRef aiirComplexTypeGetName(void) { return wrap(ComplexType::name); }

AiirType aiirComplexTypeGetElementType(AiirType type) {
  return wrap(llvm::cast<ComplexType>(unwrap(type)).getElementType());
}

//===----------------------------------------------------------------------===//
// Shaped type.
//===----------------------------------------------------------------------===//

bool aiirTypeIsAShaped(AiirType type) {
  return llvm::isa<ShapedType>(unwrap(type));
}

AiirType aiirShapedTypeGetElementType(AiirType type) {
  return wrap(llvm::cast<ShapedType>(unwrap(type)).getElementType());
}

bool aiirShapedTypeHasRank(AiirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).hasRank();
}

int64_t aiirShapedTypeGetRank(AiirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).getRank();
}

bool aiirShapedTypeHasStaticShape(AiirType type) {
  return llvm::cast<ShapedType>(unwrap(type)).hasStaticShape();
}

bool aiirShapedTypeIsDynamicDim(AiirType type, intptr_t dim) {
  return llvm::cast<ShapedType>(unwrap(type))
      .isDynamicDim(static_cast<unsigned>(dim));
}

bool aiirShapedTypeIsStaticDim(AiirType type, intptr_t dim) {
  return llvm::cast<ShapedType>(unwrap(type))
      .isStaticDim(static_cast<unsigned>(dim));
}

int64_t aiirShapedTypeGetDimSize(AiirType type, intptr_t dim) {
  return llvm::cast<ShapedType>(unwrap(type))
      .getDimSize(static_cast<unsigned>(dim));
}

int64_t aiirShapedTypeGetDynamicSize() { return ShapedType::kDynamic; }

bool aiirShapedTypeIsDynamicSize(int64_t size) {
  return ShapedType::isDynamic(size);
}

bool aiirShapedTypeIsStaticSize(int64_t size) {
  return ShapedType::isStatic(size);
}

bool aiirShapedTypeIsDynamicStrideOrOffset(int64_t val) {
  return ShapedType::isDynamic(val);
}

bool aiirShapedTypeIsStaticStrideOrOffset(int64_t val) {
  return ShapedType::isStatic(val);
}

int64_t aiirShapedTypeGetDynamicStrideOrOffset() {
  return ShapedType::kDynamic;
}

//===----------------------------------------------------------------------===//
// Vector type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirVectorTypeGetTypeID() { return wrap(VectorType::getTypeID()); }

bool aiirTypeIsAVector(AiirType type) {
  return llvm::isa<VectorType>(unwrap(type));
}

AiirType aiirVectorTypeGet(intptr_t rank, const int64_t *shape,
                           AiirType elementType) {
  return wrap(VectorType::get(llvm::ArrayRef(shape, static_cast<size_t>(rank)),
                              unwrap(elementType)));
}

AiirStringRef aiirVectorTypeGetName(void) { return wrap(VectorType::name); }

AiirType aiirVectorTypeGetChecked(AiirLocation loc, intptr_t rank,
                                  const int64_t *shape, AiirType elementType) {
  return wrap(VectorType::getChecked(
      unwrap(loc), llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType)));
}

AiirType aiirVectorTypeGetScalable(intptr_t rank, const int64_t *shape,
                                   const bool *scalable, AiirType elementType) {
  return wrap(VectorType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      llvm::ArrayRef(scalable, static_cast<size_t>(rank))));
}

AiirType aiirVectorTypeGetScalableChecked(AiirLocation loc, intptr_t rank,
                                          const int64_t *shape,
                                          const bool *scalable,
                                          AiirType elementType) {
  return wrap(VectorType::getChecked(
      unwrap(loc), llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType),
      llvm::ArrayRef(scalable, static_cast<size_t>(rank))));
}

bool aiirVectorTypeIsScalable(AiirType type) {
  return cast<VectorType>(unwrap(type)).isScalable();
}

bool aiirVectorTypeIsDimScalable(AiirType type, intptr_t dim) {
  return cast<VectorType>(unwrap(type)).getScalableDims()[dim];
}

//===----------------------------------------------------------------------===//
// Ranked / Unranked tensor type.
//===----------------------------------------------------------------------===//

bool aiirTypeIsATensor(AiirType type) {
  return llvm::isa<TensorType>(unwrap(type));
}

AiirTypeID aiirRankedTensorTypeGetTypeID() {
  return wrap(RankedTensorType::getTypeID());
}

bool aiirTypeIsARankedTensor(AiirType type) {
  return llvm::isa<RankedTensorType>(unwrap(type));
}

AiirTypeID aiirUnrankedTensorTypeGetTypeID() {
  return wrap(UnrankedTensorType::getTypeID());
}

bool aiirTypeIsAUnrankedTensor(AiirType type) {
  return llvm::isa<UnrankedTensorType>(unwrap(type));
}

AiirType aiirRankedTensorTypeGet(intptr_t rank, const int64_t *shape,
                                 AiirType elementType, AiirAttribute encoding) {
  return wrap(
      RankedTensorType::get(llvm::ArrayRef(shape, static_cast<size_t>(rank)),
                            unwrap(elementType), unwrap(encoding)));
}

AiirStringRef aiirRankedTensorTypeGetName(void) {
  return wrap(RankedTensorType::name);
}

AiirType aiirRankedTensorTypeGetChecked(AiirLocation loc, intptr_t rank,
                                        const int64_t *shape,
                                        AiirType elementType,
                                        AiirAttribute encoding) {
  return wrap(RankedTensorType::getChecked(
      unwrap(loc), llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType), unwrap(encoding)));
}

AiirAttribute aiirRankedTensorTypeGetEncoding(AiirType type) {
  return wrap(llvm::cast<RankedTensorType>(unwrap(type)).getEncoding());
}

AiirType aiirUnrankedTensorTypeGet(AiirType elementType) {
  return wrap(UnrankedTensorType::get(unwrap(elementType)));
}

AiirStringRef aiirUnrankedTensorTypeGetName(void) {
  return wrap(UnrankedTensorType::name);
}

AiirType aiirUnrankedTensorTypeGetChecked(AiirLocation loc,
                                          AiirType elementType) {
  return wrap(UnrankedTensorType::getChecked(unwrap(loc), unwrap(elementType)));
}

//===----------------------------------------------------------------------===//
// Ranked / Unranked MemRef type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirMemRefTypeGetTypeID() { return wrap(MemRefType::getTypeID()); }

bool aiirTypeIsAMemRef(AiirType type) {
  return llvm::isa<MemRefType>(unwrap(type));
}

AiirType aiirMemRefTypeGet(AiirType elementType, intptr_t rank,
                           const int64_t *shape, AiirAttribute layout,
                           AiirAttribute memorySpace) {
  return wrap(MemRefType::get(
      llvm::ArrayRef(shape, static_cast<size_t>(rank)), unwrap(elementType),
      aiirAttributeIsNull(layout)
          ? MemRefLayoutAttrInterface()
          : llvm::cast<MemRefLayoutAttrInterface>(unwrap(layout)),
      unwrap(memorySpace)));
}

AiirStringRef aiirMemRefTypeGetName(void) { return wrap(MemRefType::name); }

AiirType aiirMemRefTypeGetChecked(AiirLocation loc, AiirType elementType,
                                  intptr_t rank, const int64_t *shape,
                                  AiirAttribute layout,
                                  AiirAttribute memorySpace) {
  return wrap(MemRefType::getChecked(
      unwrap(loc), llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType),
      aiirAttributeIsNull(layout)
          ? MemRefLayoutAttrInterface()
          : llvm::cast<MemRefLayoutAttrInterface>(unwrap(layout)),
      unwrap(memorySpace)));
}

AiirType aiirMemRefTypeContiguousGet(AiirType elementType, intptr_t rank,
                                     const int64_t *shape,
                                     AiirAttribute memorySpace) {
  return wrap(MemRefType::get(llvm::ArrayRef(shape, static_cast<size_t>(rank)),
                              unwrap(elementType), MemRefLayoutAttrInterface(),
                              unwrap(memorySpace)));
}

AiirType aiirMemRefTypeContiguousGetChecked(AiirLocation loc,
                                            AiirType elementType, intptr_t rank,
                                            const int64_t *shape,
                                            AiirAttribute memorySpace) {
  return wrap(MemRefType::getChecked(
      unwrap(loc), llvm::ArrayRef(shape, static_cast<size_t>(rank)),
      unwrap(elementType), MemRefLayoutAttrInterface(), unwrap(memorySpace)));
}

AiirAttribute aiirMemRefTypeGetLayout(AiirType type) {
  return wrap(llvm::cast<MemRefType>(unwrap(type)).getLayout());
}

AiirAffineMap aiirMemRefTypeGetAffineMap(AiirType type) {
  return wrap(llvm::cast<MemRefType>(unwrap(type)).getLayout().getAffineMap());
}

AiirAttribute aiirMemRefTypeGetMemorySpace(AiirType type) {
  return wrap(llvm::cast<MemRefType>(unwrap(type)).getMemorySpace());
}

AiirLogicalResult aiirMemRefTypeGetStridesAndOffset(AiirType type,
                                                    int64_t *strides,
                                                    int64_t *offset) {
  MemRefType memrefType = llvm::cast<MemRefType>(unwrap(type));
  SmallVector<int64_t> strides_;
  if (failed(memrefType.getStridesAndOffset(strides_, *offset)))
    return aiirLogicalResultFailure();

  (void)llvm::copy(strides_, strides);
  return aiirLogicalResultSuccess();
}

AiirTypeID aiirUnrankedMemRefTypeGetTypeID() {
  return wrap(UnrankedMemRefType::getTypeID());
}

bool aiirTypeIsAUnrankedMemRef(AiirType type) {
  return llvm::isa<UnrankedMemRefType>(unwrap(type));
}

AiirType aiirUnrankedMemRefTypeGet(AiirType elementType,
                                   AiirAttribute memorySpace) {
  return wrap(
      UnrankedMemRefType::get(unwrap(elementType), unwrap(memorySpace)));
}

AiirStringRef aiirUnrankedMemRefTypeGetName(void) {
  return wrap(UnrankedMemRefType::name);
}

AiirType aiirUnrankedMemRefTypeGetChecked(AiirLocation loc,
                                          AiirType elementType,
                                          AiirAttribute memorySpace) {
  return wrap(UnrankedMemRefType::getChecked(unwrap(loc), unwrap(elementType),
                                             unwrap(memorySpace)));
}

AiirAttribute aiirUnrankedMemrefGetMemorySpace(AiirType type) {
  return wrap(llvm::cast<UnrankedMemRefType>(unwrap(type)).getMemorySpace());
}

//===----------------------------------------------------------------------===//
// Tuple type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirTupleTypeGetTypeID() { return wrap(TupleType::getTypeID()); }

bool aiirTypeIsATuple(AiirType type) {
  return llvm::isa<TupleType>(unwrap(type));
}

AiirType aiirTupleTypeGet(AiirContext ctx, intptr_t numElements,
                          AiirType const *elements) {
  SmallVector<Type, 4> types;
  ArrayRef<Type> typeRef = unwrapList(numElements, elements, types);
  return wrap(TupleType::get(unwrap(ctx), typeRef));
}

AiirStringRef aiirTupleTypeGetName(void) { return wrap(TupleType::name); }

intptr_t aiirTupleTypeGetNumTypes(AiirType type) {
  return llvm::cast<TupleType>(unwrap(type)).size();
}

AiirType aiirTupleTypeGetType(AiirType type, intptr_t pos) {
  return wrap(
      llvm::cast<TupleType>(unwrap(type)).getType(static_cast<size_t>(pos)));
}

//===----------------------------------------------------------------------===//
// Function type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirFunctionTypeGetTypeID() {
  return wrap(FunctionType::getTypeID());
}

bool aiirTypeIsAFunction(AiirType type) {
  return llvm::isa<FunctionType>(unwrap(type));
}

AiirType aiirFunctionTypeGet(AiirContext ctx, intptr_t numInputs,
                             AiirType const *inputs, intptr_t numResults,
                             AiirType const *results) {
  SmallVector<Type, 4> inputsList;
  SmallVector<Type, 4> resultsList;
  (void)unwrapList(numInputs, inputs, inputsList);
  (void)unwrapList(numResults, results, resultsList);
  return wrap(FunctionType::get(unwrap(ctx), inputsList, resultsList));
}

AiirStringRef aiirFunctionTypeGetName(void) { return wrap(FunctionType::name); }

intptr_t aiirFunctionTypeGetNumInputs(AiirType type) {
  return llvm::cast<FunctionType>(unwrap(type)).getNumInputs();
}

intptr_t aiirFunctionTypeGetNumResults(AiirType type) {
  return llvm::cast<FunctionType>(unwrap(type)).getNumResults();
}

AiirType aiirFunctionTypeGetInput(AiirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(llvm::cast<FunctionType>(unwrap(type))
                  .getInput(static_cast<unsigned>(pos)));
}

AiirType aiirFunctionTypeGetResult(AiirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(llvm::cast<FunctionType>(unwrap(type))
                  .getResult(static_cast<unsigned>(pos)));
}

//===----------------------------------------------------------------------===//
// Opaque type.
//===----------------------------------------------------------------------===//

AiirTypeID aiirOpaqueTypeGetTypeID() { return wrap(OpaqueType::getTypeID()); }

bool aiirTypeIsAOpaque(AiirType type) {
  return llvm::isa<OpaqueType>(unwrap(type));
}

AiirType aiirOpaqueTypeGet(AiirContext ctx, AiirStringRef dialectNamespace,
                           AiirStringRef typeData) {
  return wrap(
      OpaqueType::get(StringAttr::get(unwrap(ctx), unwrap(dialectNamespace)),
                      unwrap(typeData)));
}

AiirStringRef aiirOpaqueTypeGetName(void) { return wrap(OpaqueType::name); }

AiirStringRef aiirOpaqueTypeGetDialectNamespace(AiirType type) {
  return wrap(
      llvm::cast<OpaqueType>(unwrap(type)).getDialectNamespace().strref());
}

AiirStringRef aiirOpaqueTypeGetData(AiirType type) {
  return wrap(llvm::cast<OpaqueType>(unwrap(type)).getTypeData());
}
