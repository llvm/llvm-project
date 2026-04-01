//===- LLVM.cpp - C Interface for LLVM dialect ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/Dialect/LLVM.h"
#include "aiir-c/IR.h"
#include "aiir-c/Support.h"
#include "aiir/CAPI/Registration.h"
#include "aiir/CAPI/Wrap.h"
#include "aiir/Dialect/LLVMIR/LLVMAttrs.h"
#include "aiir/Dialect/LLVMIR/LLVMDialect.h"
#include "aiir/Dialect/LLVMIR/LLVMTypes.h"
#include "llvm-c/Core.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

using namespace aiir;
using namespace aiir::LLVM;

AIIR_DEFINE_CAPI_DIALECT_REGISTRATION(LLVM, llvm, LLVMDialect)

AiirType aiirLLVMPointerTypeGet(AiirContext ctx, unsigned addressSpace) {
  return wrap(LLVMPointerType::get(unwrap(ctx), addressSpace));
}

AiirStringRef aiirLLVMPointerTypeGetName(void) {
  return wrap(LLVM::LLVMPointerType::name);
}

AiirTypeID aiirLLVMPointerTypeGetTypeID() {
  return wrap(LLVM::LLVMPointerType::getTypeID());
}

bool aiirTypeIsALLVMPointerType(AiirType type) {
  return isa<LLVM::LLVMPointerType>(unwrap(type));
}

unsigned aiirLLVMPointerTypeGetAddressSpace(AiirType pointerType) {
  return cast<LLVM::LLVMPointerType>(unwrap(pointerType)).getAddressSpace();
}

AiirType aiirLLVMVoidTypeGet(AiirContext ctx) {
  return wrap(LLVMVoidType::get(unwrap(ctx)));
}

AiirStringRef aiirLLVMVoidTypeGetName(void) { return wrap(LLVMVoidType::name); }

bool aiirTypeIsALLVMArrayType(AiirType type) {
  return isa<LLVM::LLVMArrayType>(unwrap(type));
}

AiirTypeID aiirLLVMArrayTypeGetTypeID() {
  return wrap(LLVM::LLVMArrayType::getTypeID());
}

AiirType aiirLLVMArrayTypeGet(AiirType elementType, unsigned numElements) {
  return wrap(LLVMArrayType::get(unwrap(elementType), numElements));
}

AiirStringRef aiirLLVMArrayTypeGetName(void) {
  return wrap(LLVMArrayType::name);
}

AiirType aiirLLVMArrayTypeGetElementType(AiirType type) {
  return wrap(cast<LLVM::LLVMArrayType>(unwrap(type)).getElementType());
}

unsigned aiirLLVMArrayTypeGetNumElements(AiirType type) {
  return cast<LLVM::LLVMArrayType>(unwrap(type)).getNumElements();
}

AiirType aiirLLVMFunctionTypeGet(AiirType resultType, intptr_t nArgumentTypes,
                                 AiirType const *argumentTypes, bool isVarArg) {
  SmallVector<Type, 2> argumentStorage;
  return wrap(LLVMFunctionType::get(
      unwrap(resultType),
      unwrapList(nArgumentTypes, argumentTypes, argumentStorage), isVarArg));
}

AiirStringRef aiirLLVMFunctionTypeGetName(void) {
  return wrap(LLVMFunctionType::name);
}

bool aiirTypeIsALLVMFunctionType(AiirType type) {
  return isa<LLVM::LLVMFunctionType>(unwrap(type));
}

AiirTypeID aiirLLVMFunctionTypeGetTypeID(void) {
  return wrap(LLVM::LLVMFunctionType::getTypeID());
}

intptr_t aiirLLVMFunctionTypeGetNumInputs(AiirType type) {
  return llvm::cast<LLVM::LLVMFunctionType>(unwrap(type)).getNumParams();
}

AiirType aiirLLVMFunctionTypeGetInput(AiirType type, intptr_t pos) {
  assert(pos >= 0 && "pos in array must be positive");
  return wrap(llvm::cast<LLVM::LLVMFunctionType>(unwrap(type))
                  .getParamType(static_cast<unsigned>(pos)));
}

AiirType aiirLLVMFunctionTypeGetReturnType(AiirType type) {
  return wrap(llvm::cast<LLVM::LLVMFunctionType>(unwrap(type)).getReturnType());
}

bool aiirLLVMFunctionTypeIsVarArg(AiirType type) {
  return llvm::cast<LLVM::LLVMFunctionType>(unwrap(type)).isVarArg();
}

bool aiirTypeIsALLVMStructType(AiirType type) {
  return isa<LLVM::LLVMStructType>(unwrap(type));
}

AiirTypeID aiirLLVMStructTypeGetTypeID() {
  return wrap(LLVM::LLVMStructType::getTypeID());
}

AiirStringRef aiirLLVMStructTypeGetName(void) {
  return wrap(LLVM::LLVMStructType::name);
}

bool aiirLLVMStructTypeIsLiteral(AiirType type) {
  return !cast<LLVM::LLVMStructType>(unwrap(type)).isIdentified();
}

intptr_t aiirLLVMStructTypeGetNumElementTypes(AiirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).getBody().size();
}

AiirType aiirLLVMStructTypeGetElementType(AiirType type, intptr_t position) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getBody()[position]);
}

bool aiirLLVMStructTypeIsPacked(AiirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isPacked();
}

AiirStringRef aiirLLVMStructTypeGetIdentifier(AiirType type) {
  return wrap(cast<LLVM::LLVMStructType>(unwrap(type)).getName());
}

bool aiirLLVMStructTypeIsOpaque(AiirType type) {
  return cast<LLVM::LLVMStructType>(unwrap(type)).isOpaque();
}

AiirType aiirLLVMStructTypeLiteralGet(AiirContext ctx, intptr_t nFieldTypes,
                                      AiirType const *fieldTypes,
                                      bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteral(
      unwrap(ctx), unwrapList(nFieldTypes, fieldTypes, fieldStorage),
      isPacked));
}

AiirType aiirLLVMStructTypeLiteralGetChecked(AiirLocation loc,
                                             intptr_t nFieldTypes,
                                             AiirType const *fieldTypes,
                                             bool isPacked) {
  SmallVector<Type> fieldStorage;
  return wrap(LLVMStructType::getLiteralChecked(
      [loc]() { return emitError(unwrap(loc)); }, unwrap(loc)->getContext(),
      unwrapList(nFieldTypes, fieldTypes, fieldStorage), isPacked));
}

AiirType aiirLLVMStructTypeOpaqueGet(AiirContext ctx, AiirStringRef name) {
  return wrap(LLVMStructType::getOpaque(unwrap(name), unwrap(ctx)));
}

AiirType aiirLLVMStructTypeIdentifiedGet(AiirContext ctx, AiirStringRef name) {
  return wrap(LLVMStructType::getIdentified(unwrap(ctx), unwrap(name)));
}

AiirType aiirLLVMStructTypeIdentifiedNewGet(AiirContext ctx, AiirStringRef name,
                                            intptr_t nFieldTypes,
                                            AiirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(LLVMStructType::getNewIdentified(
      unwrap(ctx), unwrap(name), unwrapList(nFieldTypes, fieldTypes, fields),
      isPacked));
}

AiirLogicalResult aiirLLVMStructTypeSetBody(AiirType structType,
                                            intptr_t nFieldTypes,
                                            AiirType const *fieldTypes,
                                            bool isPacked) {
  SmallVector<Type> fields;
  return wrap(
      cast<LLVM::LLVMStructType>(unwrap(structType))
          .setBody(unwrapList(nFieldTypes, fieldTypes, fields), isPacked));
}

AiirAttribute aiirLLVMDIExpressionElemAttrGet(AiirContext ctx,
                                              unsigned int opcode,
                                              intptr_t nArguments,
                                              uint64_t const *arguments) {
  auto list = ArrayRef<uint64_t>(arguments, nArguments);
  return wrap(DIExpressionElemAttr::get(unwrap(ctx), opcode, list));
}

AiirStringRef aiirLLVMDIExpressionElemAttrGetName(void) {
  return wrap(DIExpressionElemAttr::name);
}

AiirAttribute aiirLLVMDIExpressionAttrGet(AiirContext ctx, intptr_t nOperations,
                                          AiirAttribute const *operations) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nOperations);

  return wrap(DIExpressionAttr::get(
      unwrap(ctx),
      llvm::map_to_vector(unwrapList(nOperations, operations, attrStorage),
                          llvm::CastTo<DIExpressionElemAttr>)));
}

AiirStringRef aiirLLVMDIExpressionAttrGetName(void) {
  return wrap(DIExpressionAttr::name);
}

AiirAttribute aiirLLVMDINullTypeAttrGet(AiirContext ctx) {
  return wrap(DINullTypeAttr::get(unwrap(ctx)));
}

AiirStringRef aiirLLVMDINullTypeAttrGetName(void) {
  return wrap(DINullTypeAttr::name);
}

AiirAttribute aiirLLVMDIBasicTypeAttrGet(AiirContext ctx, unsigned int tag,
                                         AiirAttribute name,
                                         uint64_t sizeInBits,
                                         AiirLLVMTypeEncoding encoding) {

  return wrap(DIBasicTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, encoding));
}

AiirStringRef aiirLLVMDIBasicTypeAttrGetName(void) {
  return wrap(DIBasicTypeAttr::name);
}

AiirAttribute aiirLLVMDICompositeTypeAttrGetRecSelf(AiirAttribute recId) {
  return wrap(
      DICompositeTypeAttr::getRecSelf(cast<DistinctAttr>(unwrap(recId))));
}

AiirAttribute aiirLLVMDICompositeTypeAttrGet(
    AiirContext ctx, AiirAttribute recId, bool isRecSelf, unsigned int tag,
    AiirAttribute name, AiirAttribute file, uint32_t line, AiirAttribute scope,
    AiirAttribute baseType, int64_t flags, uint64_t sizeInBits,
    uint64_t alignInBits, intptr_t nElements, AiirAttribute const *elements,
    AiirAttribute dataLocation, AiirAttribute rank, AiirAttribute allocated,
    AiirAttribute associated) {
  SmallVector<Attribute> elementsStorage;
  elementsStorage.reserve(nElements);

  return wrap(DICompositeTypeAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(recId)), isRecSelf, tag,
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(file)), line,
      cast<DIScopeAttr>(unwrap(scope)), cast<DITypeAttr>(unwrap(baseType)),
      DIFlags(flags), sizeInBits, alignInBits,
      cast<DIExpressionAttr>(unwrap(dataLocation)),
      cast<DIExpressionAttr>(unwrap(rank)),
      cast<DIExpressionAttr>(unwrap(allocated)),
      cast<DIExpressionAttr>(unwrap(associated)),
      llvm::map_to_vector(unwrapList(nElements, elements, elementsStorage),
                          llvm::CastTo<DINodeAttr>)));
}

AiirStringRef aiirLLVMDICompositeTypeAttrGetName(void) {
  return wrap(DICompositeTypeAttr::name);
}

AiirAttribute aiirLLVMDIDerivedTypeAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute name, AiirAttribute file,
    uint32_t line, AiirAttribute scope, AiirAttribute baseType,
    uint64_t sizeInBits, uint32_t alignInBits, uint64_t offsetInBits,
    int64_t dwarfAddressSpace, int64_t flags, AiirAttribute extraData) {
  std::optional<unsigned> addressSpace = std::nullopt;
  if (dwarfAddressSpace >= 0)
    addressSpace = (unsigned)dwarfAddressSpace;
  return wrap(DIDerivedTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)),
      cast<DIFileAttr>(unwrap(file)), line, cast<DIScopeAttr>(unwrap(scope)),
      cast<DITypeAttr>(unwrap(baseType)), sizeInBits, alignInBits, offsetInBits,
      addressSpace, DIFlags(flags), cast<DINodeAttr>(unwrap(extraData))));
}

AiirStringRef aiirLLVMDIDerivedTypeAttrGetName(void) {
  return wrap(DIDerivedTypeAttr::name);
}

AiirAttribute aiirLLVMDIStringTypeAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute name, uint64_t sizeInBits,
    uint32_t alignInBits, AiirAttribute stringLength,
    AiirAttribute stringLengthExp, AiirAttribute stringLocationExp,
    AiirLLVMTypeEncoding encoding) {
  return wrap(DIStringTypeAttr::get(
      unwrap(ctx), tag, cast<StringAttr>(unwrap(name)), sizeInBits, alignInBits,
      cast<DIVariableAttr>(unwrap(stringLength)),
      cast<DIExpressionAttr>(unwrap(stringLengthExp)),
      cast<DIExpressionAttr>(unwrap(stringLocationExp)), encoding));
}

AiirStringRef aiirLLVMDIStringTypeAttrGetName(void) {
  return wrap(DIStringTypeAttr::name);
}

AiirAttribute
aiirLLVMDIDerivedTypeAttrGetBaseType(AiirAttribute diDerivedType) {
  return wrap(cast<DIDerivedTypeAttr>(unwrap(diDerivedType)).getBaseType());
}

AiirAttribute aiirLLVMCConvAttrGet(AiirContext ctx, AiirLLVMCConv cconv) {
  return wrap(CConvAttr::get(unwrap(ctx), CConv(cconv)));
}

AiirStringRef aiirLLVMCConvAttrGetName(void) { return wrap(CConvAttr::name); }

AiirAttribute aiirLLVMComdatAttrGet(AiirContext ctx, AiirLLVMComdat comdat) {
  return wrap(ComdatAttr::get(unwrap(ctx), comdat::Comdat(comdat)));
}

AiirStringRef aiirLLVMComdatAttrGetName(void) { return wrap(ComdatAttr::name); }

AiirAttribute aiirLLVMLinkageAttrGet(AiirContext ctx, AiirLLVMLinkage linkage) {
  return wrap(LinkageAttr::get(unwrap(ctx), linkage::Linkage(linkage)));
}

AiirStringRef aiirLLVMLinkageAttrGetName(void) {
  return wrap(LinkageAttr::name);
}

AiirAttribute aiirLLVMDIFileAttrGet(AiirContext ctx, AiirAttribute name,
                                    AiirAttribute directory) {
  return wrap(DIFileAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                              cast<StringAttr>(unwrap(directory))));
}

AiirStringRef aiirLLVMDIFileAttrGetName(void) { return wrap(DIFileAttr::name); }

AiirAttribute aiirLLVMDICompileUnitAttrGet(
    AiirContext ctx, AiirAttribute id, unsigned int sourceLanguage,
    AiirAttribute file, AiirAttribute producer, bool isOptimized,
    AiirLLVMDIEmissionKind emissionKind, bool isDebugInfoForProfiling,
    AiirLLVMDINameTableKind nameTableKind, AiirAttribute splitDebugFilename,
    intptr_t nImportedEntities, AiirAttribute const *importedEntities) {
  SmallVector<Attribute> importsStorage;
  importsStorage.reserve(nImportedEntities);
  return wrap(DICompileUnitAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(id)), sourceLanguage,
      cast<DIFileAttr>(unwrap(file)), cast<StringAttr>(unwrap(producer)),
      isOptimized, DIEmissionKind(emissionKind), isDebugInfoForProfiling,
      DINameTableKind(nameTableKind),
      cast<StringAttr>(unwrap(splitDebugFilename)),
      llvm::map_to_vector(
          unwrapList(nImportedEntities, importedEntities, importsStorage),
          llvm::CastTo<DINodeAttr>)));
}

AiirStringRef aiirLLVMDICompileUnitAttrGetName(void) {
  return wrap(DICompileUnitAttr::name);
}

AiirAttribute aiirLLVMDIFlagsAttrGet(AiirContext ctx, uint64_t value) {
  return wrap(DIFlagsAttr::get(unwrap(ctx), DIFlags(value)));
}

AiirStringRef aiirLLVMDIFlagsAttrGetName(void) {
  return wrap(DIFlagsAttr::name);
}

AiirAttribute aiirLLVMDILexicalBlockAttrGet(AiirContext ctx,
                                            AiirAttribute scope,
                                            AiirAttribute file,
                                            unsigned int line,
                                            unsigned int column) {
  return wrap(
      DILexicalBlockAttr::get(unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
                              cast<DIFileAttr>(unwrap(file)), line, column));
}

AiirStringRef aiirLLVMDILexicalBlockAttrGetName(void) {
  return wrap(DILexicalBlockAttr::name);
}

AiirAttribute aiirLLVMDILexicalBlockFileAttrGet(AiirContext ctx,
                                                AiirAttribute scope,
                                                AiirAttribute file,
                                                unsigned int discriminator) {
  return wrap(DILexicalBlockFileAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<DIFileAttr>(unwrap(file)), discriminator));
}

AiirStringRef aiirLLVMDILexicalBlockFileAttrGetName(void) {
  return wrap(DILexicalBlockFileAttr::name);
}

AiirAttribute aiirLLVMDILocalVariableAttrGet(
    AiirContext ctx, AiirAttribute scope, AiirAttribute name,
    AiirAttribute diFile, unsigned int line, unsigned int arg,
    unsigned int alignInBits, AiirAttribute diType, int64_t flags) {
  return wrap(DILocalVariableAttr::get(
      unwrap(ctx), cast<DIScopeAttr>(unwrap(scope)),
      cast<StringAttr>(unwrap(name)), cast<DIFileAttr>(unwrap(diFile)), line,
      arg, alignInBits, cast<DITypeAttr>(unwrap(diType)), DIFlags(flags)));
}

AiirStringRef aiirLLVMDILocalVariableAttrGetName(void) {
  return wrap(DILocalVariableAttr::name);
}

AiirAttribute aiirLLVMDISubroutineTypeAttrGet(AiirContext ctx,
                                              unsigned int callingConvention,
                                              intptr_t nTypes,
                                              AiirAttribute const *types) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nTypes);

  return wrap(DISubroutineTypeAttr::get(
      unwrap(ctx), callingConvention,
      llvm::map_to_vector(unwrapList(nTypes, types, attrStorage),
                          llvm::CastTo<DITypeAttr>)));
}

AiirStringRef aiirLLVMDISubroutineTypeAttrGetName(void) {
  return wrap(DISubroutineTypeAttr::name);
}

AiirAttribute aiirLLVMDISubprogramAttrGetRecSelf(AiirAttribute recId) {
  return wrap(DISubprogramAttr::getRecSelf(cast<DistinctAttr>(unwrap(recId))));
}

AiirAttribute aiirLLVMDISubprogramAttrGet(
    AiirContext ctx, AiirAttribute recId, bool isRecSelf, AiirAttribute id,
    AiirAttribute compileUnit, AiirAttribute scope, AiirAttribute name,
    AiirAttribute linkageName, AiirAttribute file, unsigned int line,
    unsigned int scopeLine, uint64_t subprogramFlags, AiirAttribute type,
    intptr_t nRetainedNodes, AiirAttribute const *retainedNodes,
    intptr_t nAnnotations, AiirAttribute const *annotations) {
  SmallVector<Attribute> nodesStorage;
  nodesStorage.reserve(nRetainedNodes);

  SmallVector<Attribute> annotationsStorage;
  annotationsStorage.reserve(nAnnotations);

  return wrap(DISubprogramAttr::get(
      unwrap(ctx), cast<DistinctAttr>(unwrap(recId)), isRecSelf,
      cast<DistinctAttr>(unwrap(id)),
      cast<DICompileUnitAttr>(unwrap(compileUnit)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(linkageName)), cast<DIFileAttr>(unwrap(file)),
      line, scopeLine, DISubprogramFlags(subprogramFlags),
      cast<DISubroutineTypeAttr>(unwrap(type)),
      llvm::map_to_vector(
          unwrapList(nRetainedNodes, retainedNodes, nodesStorage),
          llvm::CastTo<DINodeAttr>),
      llvm::map_to_vector(
          unwrapList(nAnnotations, annotations, annotationsStorage),
          llvm::CastTo<DINodeAttr>)));
}

AiirStringRef aiirLLVMDISubprogramAttrGetName(void) {
  return wrap(DISubprogramAttr::name);
}

AiirAttribute aiirLLVMDISubprogramAttrGetScope(AiirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getScope());
}

unsigned int aiirLLVMDISubprogramAttrGetLine(AiirAttribute diSubprogram) {
  return cast<DISubprogramAttr>(unwrap(diSubprogram)).getLine();
}

unsigned int aiirLLVMDISubprogramAttrGetScopeLine(AiirAttribute diSubprogram) {
  return cast<DISubprogramAttr>(unwrap(diSubprogram)).getScopeLine();
}

AiirAttribute
aiirLLVMDISubprogramAttrGetCompileUnit(AiirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getCompileUnit());
}

AiirAttribute aiirLLVMDISubprogramAttrGetFile(AiirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getFile());
}

AiirAttribute aiirLLVMDISubprogramAttrGetType(AiirAttribute diSubprogram) {
  return wrap(cast<DISubprogramAttr>(unwrap(diSubprogram)).getType());
}

AiirAttribute aiirLLVMDIModuleAttrGet(AiirContext ctx, AiirAttribute file,
                                      AiirAttribute scope, AiirAttribute name,
                                      AiirAttribute configMacros,
                                      AiirAttribute includePath,
                                      AiirAttribute apinotes, unsigned int line,
                                      bool isDecl) {
  return wrap(DIModuleAttr::get(
      unwrap(ctx), cast<DIFileAttr>(unwrap(file)),
      cast<DIScopeAttr>(unwrap(scope)), cast<StringAttr>(unwrap(name)),
      cast<StringAttr>(unwrap(configMacros)),
      cast<StringAttr>(unwrap(includePath)), cast<StringAttr>(unwrap(apinotes)),
      line, isDecl));
}

AiirStringRef aiirLLVMDIModuleAttrGetName(void) {
  return wrap(DIModuleAttr::name);
}

AiirAttribute aiirLLVMDIModuleAttrGetScope(AiirAttribute diModule) {
  return wrap(cast<DIModuleAttr>(unwrap(diModule)).getScope());
}

AiirAttribute aiirLLVMDIImportedEntityAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute scope,
    AiirAttribute entity, AiirAttribute file, unsigned int line,
    AiirAttribute name, intptr_t nElements, AiirAttribute const *elements) {
  SmallVector<Attribute> elementsStorage;
  elementsStorage.reserve(nElements);
  return wrap(DIImportedEntityAttr::get(
      unwrap(ctx), tag, cast<DIScopeAttr>(unwrap(scope)),
      cast<DINodeAttr>(unwrap(entity)), cast<DIFileAttr>(unwrap(file)), line,
      cast<StringAttr>(unwrap(name)),
      llvm::map_to_vector(unwrapList(nElements, elements, elementsStorage),
                          llvm::CastTo<DINodeAttr>)));
}

AiirStringRef aiirLLVMDIImportedEntityAttrGetName(void) {
  return wrap(DIImportedEntityAttr::name);
}

AiirAttribute aiirLLVMDIAnnotationAttrGet(AiirContext ctx, AiirAttribute name,
                                          AiirAttribute value) {
  return wrap(DIAnnotationAttr::get(unwrap(ctx), cast<StringAttr>(unwrap(name)),
                                    cast<StringAttr>(unwrap(value))));
}

AiirStringRef aiirLLVMDIAnnotationAttrGetName(void) {
  return wrap(DIAnnotationAttr::name);
}

//===----------------------------------------------------------------------===//
// Metadata Attributes
//===----------------------------------------------------------------------===//

AiirAttribute aiirLLVMMDStringAttrGet(AiirContext ctx, AiirStringRef value) {
  return wrap(MDStringAttr::get(unwrap(ctx),
                                StringAttr::get(unwrap(ctx), unwrap(value))));
}

bool aiirLLVMAttrIsAMDStringAttr(AiirAttribute attr) {
  return isa<MDStringAttr>(unwrap(attr));
}

AiirTypeID aiirLLVMMDStringAttrGetTypeID(void) {
  return wrap(MDStringAttr::getTypeID());
}

AiirStringRef aiirLLVMMDStringAttrGetValue(AiirAttribute attr) {
  return wrap(cast<MDStringAttr>(unwrap(attr)).getValue().getValue());
}

AiirAttribute aiirLLVMMDConstantAttrGet(AiirContext ctx,
                                        AiirAttribute valueAttr) {
  return wrap(MDConstantAttr::get(unwrap(ctx), unwrap(valueAttr)));
}

bool aiirLLVMAttrIsAMDConstantAttr(AiirAttribute attr) {
  return isa<MDConstantAttr>(unwrap(attr));
}

AiirTypeID aiirLLVMMDConstantAttrGetTypeID(void) {
  return wrap(MDConstantAttr::getTypeID());
}

AiirAttribute aiirLLVMMDConstantAttrGetValue(AiirAttribute attr) {
  return wrap((Attribute)cast<MDConstantAttr>(unwrap(attr)).getValue());
}

AiirAttribute aiirLLVMMDFuncAttrGet(AiirContext ctx, AiirAttribute name) {
  return wrap(
      MDFuncAttr::get(unwrap(ctx), cast<FlatSymbolRefAttr>(unwrap(name))));
}

bool aiirLLVMAttrIsAMDFuncAttr(AiirAttribute attr) {
  return isa<MDFuncAttr>(unwrap(attr));
}

AiirTypeID aiirLLVMMDFuncAttrGetTypeID(void) {
  return wrap(MDFuncAttr::getTypeID());
}

AiirAttribute aiirLLVMMDFuncAttrGetName(AiirAttribute attr) {
  return wrap((Attribute)cast<MDFuncAttr>(unwrap(attr)).getName());
}

AiirAttribute aiirLLVMMDNodeAttrGet(AiirContext ctx, intptr_t nOperands,
                                    AiirAttribute const *operands) {
  SmallVector<Attribute> attrStorage;
  attrStorage.reserve(nOperands);
  return wrap(MDNodeAttr::get(unwrap(ctx),
                              unwrapList(nOperands, operands, attrStorage)));
}

bool aiirLLVMAttrIsAMDNodeAttr(AiirAttribute attr) {
  return isa<MDNodeAttr>(unwrap(attr));
}

AiirTypeID aiirLLVMMDNodeAttrGetTypeID(void) {
  return wrap(MDNodeAttr::getTypeID());
}

intptr_t aiirLLVMMDNodeAttrGetNumOperands(AiirAttribute attr) {
  return cast<MDNodeAttr>(unwrap(attr)).getOperands().size();
}

AiirAttribute aiirLLVMMDNodeAttrGetOperand(AiirAttribute attr, intptr_t index) {
  return wrap(cast<MDNodeAttr>(unwrap(attr)).getOperands()[index]);
}
