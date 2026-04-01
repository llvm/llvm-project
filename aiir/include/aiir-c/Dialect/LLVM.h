//===-- aiir-c/Dialect/LLVM.h - C API for LLVM --------------------*- C -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_DIALECT_LLVM_H
#define AIIR_C_DIALECT_LLVM_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

AIIR_DECLARE_CAPI_DIALECT_REGISTRATION(LLVM, llvm);

/// Creates an llvm.ptr type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMPointerTypeGet(AiirContext ctx,
                                                   unsigned addressSpace);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMPointerTypeGetName(void);

AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMPointerTypeGetTypeID(void);

/// Returns `true` if the type is an LLVM dialect pointer type.
AIIR_CAPI_EXPORTED bool aiirTypeIsALLVMPointerType(AiirType type);

/// Returns address space of llvm.ptr
AIIR_CAPI_EXPORTED unsigned
aiirLLVMPointerTypeGetAddressSpace(AiirType pointerType);

/// Creates an llmv.void type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMVoidTypeGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMVoidTypeGetName(void);

/// Returns `true` if the type is an LLVM dialect array type.
AIIR_CAPI_EXPORTED bool aiirTypeIsALLVMArrayType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMArrayTypeGetTypeID(void);

/// Creates an llvm.array type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMArrayTypeGet(AiirType elementType,
                                                 unsigned numElements);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMArrayTypeGetName(void);

/// Returns the element type of the llvm.array type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMArrayTypeGetElementType(AiirType type);

/// Returns the number of elements in the llvm.array type.
AIIR_CAPI_EXPORTED unsigned aiirLLVMArrayTypeGetNumElements(AiirType type);

/// Creates an llvm.func type.
AIIR_CAPI_EXPORTED AiirType
aiirLLVMFunctionTypeGet(AiirType resultType, intptr_t nArgumentTypes,
                        AiirType const *argumentTypes, bool isVarArg);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMFunctionTypeGetName(void);

/// Returns `true` if the type is an LLVM dialect function type.
AIIR_CAPI_EXPORTED bool aiirTypeIsALLVMFunctionType(AiirType type);

/// Returns the TypeID of an LLVM function type.
AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMFunctionTypeGetTypeID(void);

/// Returns the number of input types.
AIIR_CAPI_EXPORTED intptr_t aiirLLVMFunctionTypeGetNumInputs(AiirType type);

/// Returns the pos-th input type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMFunctionTypeGetInput(AiirType type,
                                                         intptr_t pos);

/// Returns `true` if the function type is variadic.
AIIR_CAPI_EXPORTED bool aiirLLVMFunctionTypeIsVarArg(AiirType type);

/// Returns the return type of the function type.
AIIR_CAPI_EXPORTED AiirType aiirLLVMFunctionTypeGetReturnType(AiirType type);

/// Returns `true` if the type is an LLVM dialect struct type.
AIIR_CAPI_EXPORTED bool aiirTypeIsALLVMStructType(AiirType type);

AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMStructTypeGetTypeID(void);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMStructTypeGetName(void);

/// Returns `true` if the type is a literal (unnamed) LLVM struct type.
AIIR_CAPI_EXPORTED bool aiirLLVMStructTypeIsLiteral(AiirType type);

/// Returns the number of fields in the struct. Asserts if the struct is opaque
/// or not yet initialized.
AIIR_CAPI_EXPORTED intptr_t aiirLLVMStructTypeGetNumElementTypes(AiirType type);

/// Returns the `positions`-th field of the struct. Asserts if the struct is
/// opaque, not yet initialized or if the position is out of range.
AIIR_CAPI_EXPORTED AiirType aiirLLVMStructTypeGetElementType(AiirType type,
                                                             intptr_t position);

/// Returns `true` if the struct is packed.
AIIR_CAPI_EXPORTED bool aiirLLVMStructTypeIsPacked(AiirType type);

/// Returns the identifier of the identified struct. Asserts that the struct is
/// identified, i.e., not literal.
AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMStructTypeGetIdentifier(AiirType type);

/// Returns `true` is the struct is explicitly opaque (will not have a body) or
/// uninitialized (will eventually have a body).
AIIR_CAPI_EXPORTED bool aiirLLVMStructTypeIsOpaque(AiirType type);

/// Creates an LLVM literal (unnamed) struct type. This may assert if the fields
/// have types not compatible with the LLVM dialect. For a graceful failure, use
/// the checked version.
AIIR_CAPI_EXPORTED AiirType
aiirLLVMStructTypeLiteralGet(AiirContext ctx, intptr_t nFieldTypes,
                             AiirType const *fieldTypes, bool isPacked);

/// Creates an LLVM literal (unnamed) struct type if possible. Emits a
/// diagnostic at the given location and returns null otherwise.
AIIR_CAPI_EXPORTED AiirType
aiirLLVMStructTypeLiteralGetChecked(AiirLocation loc, intptr_t nFieldTypes,
                                    AiirType const *fieldTypes, bool isPacked);

/// Creates an LLVM identified struct type with no body. If a struct type with
/// this name already exists in the context, returns that type. Use
/// aiirLLVMStructTypeIdentifiedNewGet to create a fresh struct type,
/// potentially renaming it. The body should be set separatelty by calling
/// aiirLLVMStructTypeSetBody, if it isn't set already.
AIIR_CAPI_EXPORTED AiirType aiirLLVMStructTypeIdentifiedGet(AiirContext ctx,
                                                            AiirStringRef name);

/// Creates an LLVM identified struct type with no body and a name starting with
/// the given prefix. If a struct with the exact name as the given prefix
/// already exists, appends an unspecified suffix to the name so that the name
/// is unique in context.
AIIR_CAPI_EXPORTED AiirType aiirLLVMStructTypeIdentifiedNewGet(
    AiirContext ctx, AiirStringRef name, intptr_t nFieldTypes,
    AiirType const *fieldTypes, bool isPacked);

AIIR_CAPI_EXPORTED AiirType aiirLLVMStructTypeOpaqueGet(AiirContext ctx,
                                                        AiirStringRef name);

/// Sets the body of the identified struct if it hasn't been set yet. Returns
/// whether the operation was successful.
AIIR_CAPI_EXPORTED AiirLogicalResult
aiirLLVMStructTypeSetBody(AiirType structType, intptr_t nFieldTypes,
                          AiirType const *fieldTypes, bool isPacked);

enum AiirLLVMCConv {
  AiirLLVMCConvC = 0,
  AiirLLVMCConvFast = 8,
  AiirLLVMCConvCold = 9,
  AiirLLVMCConvGHC = 10,
  AiirLLVMCConvHiPE = 11,
  AiirLLVMCConvAnyReg = 13,
  AiirLLVMCConvPreserveMost = 14,
  AiirLLVMCConvPreserveAll = 15,
  AiirLLVMCConvSwift = 16,
  AiirLLVMCConvCXX_FAST_TLS = 17,
  AiirLLVMCConvTail = 18,
  AiirLLVMCConvCFGuard_Check = 19,
  AiirLLVMCConvSwiftTail = 20,
  AiirLLVMCConvX86_StdCall = 64,
  AiirLLVMCConvX86_FastCall = 65,
  AiirLLVMCConvARM_APCS = 66,
  AiirLLVMCConvARM_AAPCS = 67,
  AiirLLVMCConvARM_AAPCS_VFP = 68,
  AiirLLVMCConvMSP430_INTR = 69,
  AiirLLVMCConvX86_ThisCall = 70,
  AiirLLVMCConvPTX_Kernel = 71,
  AiirLLVMCConvPTX_Device = 72,
  AiirLLVMCConvSPIR_FUNC = 75,
  AiirLLVMCConvSPIR_KERNEL = 76,
  AiirLLVMCConvIntel_OCL_BI = 77,
  AiirLLVMCConvX86_64_SysV = 78,
  AiirLLVMCConvWin64 = 79,
  AiirLLVMCConvX86_VectorCall = 80,
  AiirLLVMCConvDUMMY_HHVM = 81,
  AiirLLVMCConvDUMMY_HHVM_C = 82,
  AiirLLVMCConvX86_INTR = 83,
  AiirLLVMCConvAVR_INTR = 84,
  AiirLLVMCConvAVR_BUILTIN = 86,
  AiirLLVMCConvAMDGPU_VS = 87,
  AiirLLVMCConvAMDGPU_GS = 88,
  AiirLLVMCConvAMDGPU_CS = 90,
  AiirLLVMCConvAMDGPU_KERNEL = 91,
  AiirLLVMCConvX86_RegCall = 92,
  AiirLLVMCConvAMDGPU_HS = 93,
  AiirLLVMCConvMSP430_BUILTIN = 94,
  AiirLLVMCConvAMDGPU_LS = 95,
  AiirLLVMCConvAMDGPU_ES = 96,
  AiirLLVMCConvAArch64_VectorCall = 97,
  AiirLLVMCConvAArch64_SVE_VectorCall = 98,
  AiirLLVMCConvWASM_EmscriptenInvoke = 99,
  AiirLLVMCConvAMDGPU_Gfx = 100,
  AiirLLVMCConvM68k_INTR = 101,
};

typedef enum AiirLLVMCConv AiirLLVMCConv;

/// Creates a LLVM CConv attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMCConvAttrGet(AiirContext ctx,
                                                      AiirLLVMCConv cconv);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMCConvAttrGetName(void);

enum AiirLLVMComdat {
  AiirLLVMComdatAny = 0,
  AiirLLVMComdatExactMatch = 1,
  AiirLLVMComdatLargest = 2,
  AiirLLVMComdatNoDeduplicate = 3,
  AiirLLVMComdatSameSize = 4,
};

typedef enum AiirLLVMComdat AiirLLVMComdat;

/// Creates a LLVM Comdat attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMComdatAttrGet(AiirContext ctx,
                                                       AiirLLVMComdat comdat);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMComdatAttrGetName(void);

enum AiirLLVMLinkage {
  AiirLLVMLinkageExternal = 0,
  AiirLLVMLinkageAvailableExternally = 1,
  AiirLLVMLinkageLinkonce = 2,
  AiirLLVMLinkageLinkonceODR = 3,
  AiirLLVMLinkageWeak = 4,
  AiirLLVMLinkageWeakODR = 5,
  AiirLLVMLinkageAppending = 6,
  AiirLLVMLinkageInternal = 7,
  AiirLLVMLinkagePrivate = 8,
  AiirLLVMLinkageExternWeak = 9,
  AiirLLVMLinkageCommon = 10,
};

typedef enum AiirLLVMLinkage AiirLLVMLinkage;

/// Creates a LLVM Linkage attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMLinkageAttrGet(AiirContext ctx, AiirLLVMLinkage linkage);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMLinkageAttrGetName(void);

/// Creates a LLVM DINullType attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDINullTypeAttrGet(AiirContext ctx);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDINullTypeAttrGetName(void);

/// Creates a LLVM DIExpressionElem attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDIExpressionElemAttrGet(AiirContext ctx, unsigned int opcode,
                                intptr_t nArguments, uint64_t const *arguments);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIExpressionElemAttrGetName(void);

/// Creates a LLVM DIExpression attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIExpressionAttrGet(
    AiirContext ctx, intptr_t nOperations, AiirAttribute const *operations);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIExpressionAttrGetName(void);

enum AiirLLVMTypeEncoding {
  AiirLLVMTypeEncodingAddress = 0x1,
  AiirLLVMTypeEncodingBoolean = 0x2,
  AiirLLVMTypeEncodingComplexFloat = 0x31,
  AiirLLVMTypeEncodingFloatT = 0x4,
  AiirLLVMTypeEncodingSigned = 0x5,
  AiirLLVMTypeEncodingSignedChar = 0x6,
  AiirLLVMTypeEncodingUnsigned = 0x7,
  AiirLLVMTypeEncodingUnsignedChar = 0x08,
  AiirLLVMTypeEncodingImaginaryFloat = 0x09,
  AiirLLVMTypeEncodingPackedDecimal = 0x0a,
  AiirLLVMTypeEncodingNumericString = 0x0b,
  AiirLLVMTypeEncodingEdited = 0x0c,
  AiirLLVMTypeEncodingSignedFixed = 0x0d,
  AiirLLVMTypeEncodingUnsignedFixed = 0x0e,
  AiirLLVMTypeEncodingDecimalFloat = 0x0f,
  AiirLLVMTypeEncodingUTF = 0x10,
  AiirLLVMTypeEncodingUCS = 0x11,
  AiirLLVMTypeEncodingASCII = 0x12,
  AiirLLVMTypeEncodingLoUser = 0x80,
  AiirLLVMTypeEncodingHiUser = 0xff,
};

typedef enum AiirLLVMTypeEncoding AiirLLVMTypeEncoding;

/// Creates a LLVM DIBasicType attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIBasicTypeAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute name, uint64_t sizeInBits,
    AiirLLVMTypeEncoding encoding);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIBasicTypeAttrGetName(void);

/// Creates a self-referencing LLVM DICompositeType attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDICompositeTypeAttrGetRecSelf(AiirAttribute recId);

/// Creates a LLVM DICompositeType attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDICompositeTypeAttrGet(
    AiirContext ctx, AiirAttribute recId, bool isRecSelf, unsigned int tag,
    AiirAttribute name, AiirAttribute file, uint32_t line, AiirAttribute scope,
    AiirAttribute baseType, int64_t flags, uint64_t sizeInBits,
    uint64_t alignInBits, intptr_t nElements, AiirAttribute const *elements,
    AiirAttribute dataLocation, AiirAttribute rank, AiirAttribute allocated,
    AiirAttribute associated);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDICompositeTypeAttrGetName(void);

/// Creates a LLVM DIDerivedType attribute.  Note that `dwarfAddressSpace` is an
/// optional field, where `AIIR_CAPI_DWARF_ADDRESS_SPACE_NULL` indicates null
/// and non-negative values indicate a value present.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIDerivedTypeAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute name, AiirAttribute file,
    uint32_t line, AiirAttribute scope, AiirAttribute baseType,
    uint64_t sizeInBits, uint32_t alignInBits, uint64_t offsetInBits,
    int64_t dwarfAddressSpace, int64_t flags, AiirAttribute extraData);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIDerivedTypeAttrGetName(void);

AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIStringTypeAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute name, uint64_t sizeInBits,
    uint32_t alignInBits, AiirAttribute stringLength,
    AiirAttribute stringLengthExp, AiirAttribute stringLocationExp,
    AiirLLVMTypeEncoding encoding);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIStringTypeAttrGetName(void);

/// Constant to represent std::nullopt for dwarfAddressSpace to omit the field.
#define AIIR_CAPI_DWARF_ADDRESS_SPACE_NULL -1

/// Gets the base type from a LLVM DIDerivedType attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDIDerivedTypeAttrGetBaseType(AiirAttribute diDerivedType);

/// Creates a LLVM DIFileAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIFileAttrGet(AiirContext ctx,
                                                       AiirAttribute name,
                                                       AiirAttribute directory);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIFileAttrGetName(void);

enum AiirLLVMDIEmissionKind {
  AiirLLVMDIEmissionKindNone = 0,
  AiirLLVMDIEmissionKindFull = 1,
  AiirLLVMDIEmissionKindLineTablesOnly = 2,
  AiirLLVMDIEmissionKindDebugDirectivesOnly = 3,
};

typedef enum AiirLLVMDIEmissionKind AiirLLVMDIEmissionKind;

enum AiirLLVMDINameTableKind {
  AiirLLVMDINameTableKindDefault = 0,
  AiirLLVMDINameTableKindGNU = 1,
  AiirLLVMDINameTableKindNone = 2,
  AiirLLVMDINameTableKindApple = 3,
};

typedef enum AiirLLVMDINameTableKind AiirLLVMDINameTableKind;

/// Creates a LLVM DICompileUnit attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDICompileUnitAttrGet(
    AiirContext ctx, AiirAttribute id, unsigned int sourceLanguage,
    AiirAttribute file, AiirAttribute producer, bool isOptimized,
    AiirLLVMDIEmissionKind emissionKind, bool isDebugInfoForProfiling,
    AiirLLVMDINameTableKind nameTableKind, AiirAttribute splitDebugFilename,
    intptr_t nImportedEntities, AiirAttribute const *importedEntities);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDICompileUnitAttrGetName(void);

/// Creates a LLVM DIFlags attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIFlagsAttrGet(AiirContext ctx,
                                                        uint64_t value);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIFlagsAttrGetName(void);

/// Creates a LLVM DILexicalBlock attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDILexicalBlockAttrGet(
    AiirContext ctx, AiirAttribute scope, AiirAttribute file, unsigned int line,
    unsigned int column);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDILexicalBlockAttrGetName(void);

/// Creates a LLVM DILexicalBlockFile attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDILexicalBlockFileAttrGet(
    AiirContext ctx, AiirAttribute scope, AiirAttribute file,
    unsigned int discriminator);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDILexicalBlockFileAttrGetName(void);

/// Creates a LLVM DILocalVariableAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDILocalVariableAttrGet(
    AiirContext ctx, AiirAttribute scope, AiirAttribute name,
    AiirAttribute diFile, unsigned int line, unsigned int arg,
    unsigned int alignInBits, AiirAttribute diType, int64_t flags);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDILocalVariableAttrGetName(void);

/// Creates a self-referencing LLVM DISubprogramAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubprogramAttrGetRecSelf(AiirAttribute recId);

/// Creates a LLVM DISubprogramAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDISubprogramAttrGet(
    AiirContext ctx, AiirAttribute recId, bool isRecSelf, AiirAttribute id,
    AiirAttribute compileUnit, AiirAttribute scope, AiirAttribute name,
    AiirAttribute linkageName, AiirAttribute file, unsigned int line,
    unsigned int scopeLine, uint64_t subprogramFlags, AiirAttribute type,
    intptr_t nRetainedNodes, AiirAttribute const *retainedNodes,
    intptr_t nAnnotations, AiirAttribute const *annotations);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDISubprogramAttrGetName(void);

/// Creates a LLVM DIAnnotation attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIAnnotationAttrGet(
    AiirContext ctx, AiirAttribute name, AiirAttribute value);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIAnnotationAttrGetName(void);

/// Gets the scope from this DISubprogramAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubprogramAttrGetScope(AiirAttribute diSubprogram);

/// Gets the line from this DISubprogramAttr.
AIIR_CAPI_EXPORTED unsigned int
aiirLLVMDISubprogramAttrGetLine(AiirAttribute diSubprogram);

/// Gets the scope line from this DISubprogram.
AIIR_CAPI_EXPORTED unsigned int
aiirLLVMDISubprogramAttrGetScopeLine(AiirAttribute diSubprogram);

/// Gets the compile unit from this DISubprogram.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubprogramAttrGetCompileUnit(AiirAttribute diSubprogram);

/// Gets the file from this DISubprogramAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubprogramAttrGetFile(AiirAttribute diSubprogram);

/// Gets the type from this DISubprogramAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubprogramAttrGetType(AiirAttribute diSubprogram);

/// Creates a LLVM DISubroutineTypeAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDISubroutineTypeAttrGet(AiirContext ctx, unsigned int callingConvention,
                                intptr_t nTypes, AiirAttribute const *types);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDISubroutineTypeAttrGetName(void);

/// Creates a LLVM DIModuleAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIModuleAttrGet(
    AiirContext ctx, AiirAttribute file, AiirAttribute scope,
    AiirAttribute name, AiirAttribute configMacros, AiirAttribute includePath,
    AiirAttribute apinotes, unsigned int line, bool isDecl);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIModuleAttrGetName(void);

/// Creates a LLVM DIImportedEntityAttr attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMDIImportedEntityAttrGet(
    AiirContext ctx, unsigned int tag, AiirAttribute scope,
    AiirAttribute entity, AiirAttribute file, unsigned int line,
    AiirAttribute name, intptr_t nElements, AiirAttribute const *elements);

AIIR_CAPI_EXPORTED AiirStringRef aiirLLVMDIImportedEntityAttrGetName(void);

/// Gets the scope of this DIModuleAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMDIModuleAttrGetScope(AiirAttribute diModule);

//===----------------------------------------------------------------------===//
// Metadata Attributes
//===----------------------------------------------------------------------===//

/// Creates an LLVM MDStringAttr.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMMDStringAttrGet(AiirContext ctx,
                                                         AiirStringRef value);

/// Returns `true` if the attribute is an LLVM MDStringAttr.
AIIR_CAPI_EXPORTED bool aiirLLVMAttrIsAMDStringAttr(AiirAttribute attr);

/// Returns the TypeID of MDStringAttr.
AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMMDStringAttrGetTypeID(void);

/// Returns the string value of an LLVM MDStringAttr.
AIIR_CAPI_EXPORTED AiirStringRef
aiirLLVMMDStringAttrGetValue(AiirAttribute attr);

/// Creates an LLVM MDConstantAttr wrapping an attribute.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMMDConstantAttrGet(AiirContext ctx, AiirAttribute valueAttr);

/// Returns `true` if the attribute is an LLVM MDConstantAttr.
AIIR_CAPI_EXPORTED bool aiirLLVMAttrIsAMDConstantAttr(AiirAttribute attr);

/// Returns the TypeID of MDConstantAttr.
AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMMDConstantAttrGetTypeID(void);

/// Returns the attribute value of an LLVM MDConstantAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMMDConstantAttrGetValue(AiirAttribute attr);

/// Creates an LLVM MDFuncAttr referencing a function symbol.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMMDFuncAttrGet(AiirContext ctx,
                                                       AiirAttribute name);

/// Returns `true` if the attribute is an LLVM MDFuncAttr.
AIIR_CAPI_EXPORTED bool aiirLLVMAttrIsAMDFuncAttr(AiirAttribute attr);

/// Returns the TypeID of MDFuncAttr.
AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMMDFuncAttrGetTypeID(void);

/// Returns the symbol name of an LLVM MDFuncAttr.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMMDFuncAttrGetName(AiirAttribute attr);

/// Creates an LLVM MDNodeAttr.
AIIR_CAPI_EXPORTED AiirAttribute aiirLLVMMDNodeAttrGet(
    AiirContext ctx, intptr_t nOperands, AiirAttribute const *operands);

/// Returns `true` if the attribute is an LLVM MDNodeAttr.
AIIR_CAPI_EXPORTED bool aiirLLVMAttrIsAMDNodeAttr(AiirAttribute attr);

/// Returns the TypeID of MDNodeAttr.
AIIR_CAPI_EXPORTED AiirTypeID aiirLLVMMDNodeAttrGetTypeID(void);

/// Returns the number of operands in an LLVM MDNodeAttr.
AIIR_CAPI_EXPORTED intptr_t
aiirLLVMMDNodeAttrGetNumOperands(AiirAttribute attr);

/// Returns the operand at the given index of an LLVM MDNodeAttr.
AIIR_CAPI_EXPORTED AiirAttribute
aiirLLVMMDNodeAttrGetOperand(AiirAttribute attr, intptr_t index);

#ifdef __cplusplus
}
#endif

#include "aiir/Dialect/LLVMIR/Transforms/Passes.capi.h.inc"

#endif // AIIR_C_DIALECT_LLVM_H
