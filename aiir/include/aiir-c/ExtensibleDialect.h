//===-- aiir-c/ExtensibleDialect.h - Extensible dialect APIs -----*- C -*-====//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header provides APIs for extensible dialects.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_C_EXTENSIBLEDIALECT_H
#define AIIR_C_EXTENSIBLEDIALECT_H

#include "aiir-c/IR.h"
#include "aiir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

//===----------------------------------------------------------------------===//
/// Opaque type declarations (see aiir-c/IR.h for more details).
//===----------------------------------------------------------------------===//

#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(AiirDynamicOpTrait, void);
DEFINE_C_API_STRUCT(AiirDynamicTypeDefinition, void);
DEFINE_C_API_STRUCT(AiirDynamicAttrDefinition, void);

#undef DEFINE_C_API_STRUCT

/// Attach a dynamic op trait to the given operation name.
/// Note that the operation name must be modeled by dynamic dialect and must be
/// registered.
/// The ownership of the trait will be transferred to the operation name
/// after this call.
AIIR_CAPI_EXPORTED bool
aiirDynamicOpTraitAttach(AiirDynamicOpTrait dynamicOpTrait,
                         AiirStringRef opName, AiirContext context);

/// Get the dynamic op trait that indicates the operation is a terminator.
AIIR_CAPI_EXPORTED AiirDynamicOpTrait
aiirDynamicOpTraitIsTerminatorCreate(void);

/// Get the type ID of the dynamic op trait that indicates the operation is a
/// terminator.
AIIR_CAPI_EXPORTED AiirTypeID aiirDynamicOpTraitIsTerminatorGetTypeID(void);

/// Get the dynamic op trait that indicates regions have no terminator.
AIIR_CAPI_EXPORTED AiirDynamicOpTrait
aiirDynamicOpTraitNoTerminatorCreate(void);

/// Get the type ID of the dynamic op trait that indicates regions have no
/// terminator.
AIIR_CAPI_EXPORTED AiirTypeID aiirDynamicOpTraitNoTerminatorGetTypeID(void);

/// Destroy the dynamic op trait.
AIIR_CAPI_EXPORTED void
aiirDynamicOpTraitDestroy(AiirDynamicOpTrait dynamicOpTrait);

typedef struct {
  /// Optional constructor for the user data.
  /// Set to nullptr to disable it.
  void (*construct)(void *userData);
  /// Optional destructor for the user data.
  /// Set to nullptr to disable it.
  void (*destruct)(void *userData);
  /// The callback function to verify the operation.
  AiirLogicalResult (*verifyTrait)(AiirOperation op, void *userData);
  /// The callback function to verify the operation with access to regions.
  AiirLogicalResult (*verifyRegionTrait)(AiirOperation op, void *userData);
} AiirDynamicOpTraitCallbacks;

/// Create a custom dynamic op trait with the given type ID and callbacks.
AIIR_CAPI_EXPORTED AiirDynamicOpTrait aiirDynamicOpTraitCreate(
    AiirTypeID typeID, AiirDynamicOpTraitCallbacks callbacks, void *userData);

/// Check if the given dialect is an extensible dialect.
AIIR_CAPI_EXPORTED bool aiirDialectIsAExtensibleDialect(AiirDialect dialect);

/// Look up a registered type definition by type name in the given dialect.
/// Note that the dialect must be an extensible dialect.
AIIR_CAPI_EXPORTED AiirDynamicTypeDefinition
aiirExtensibleDialectLookupTypeDefinition(AiirDialect dialect,
                                          AiirStringRef typeName);

/// Check if the given type is a dynamic type.
AIIR_CAPI_EXPORTED bool aiirTypeIsADynamicType(AiirType type);

/// Get a dynamic type by instantiating the given type definition with the
/// provided attributes.
AIIR_CAPI_EXPORTED AiirType aiirDynamicTypeGet(
    AiirDynamicTypeDefinition typeDef, AiirAttribute *attrs, intptr_t numAttrs);

/// Get the number of parameters in the given dynamic type.
AIIR_CAPI_EXPORTED intptr_t aiirDynamicTypeGetNumParams(AiirType type);

/// Get the parameter at the given index in the provided dynamic type.
AIIR_CAPI_EXPORTED AiirAttribute aiirDynamicTypeGetParam(AiirType type,
                                                         intptr_t index);

/// Get the type definition of the given dynamic type.
AIIR_CAPI_EXPORTED AiirDynamicTypeDefinition
aiirDynamicTypeGetTypeDef(AiirType type);

/// Get the type ID of a dynamic type definition.
AIIR_CAPI_EXPORTED AiirTypeID
aiirDynamicTypeDefinitionGetTypeID(AiirDynamicTypeDefinition typeDef);

/// Get the name of the given dynamic type definition.
AIIR_CAPI_EXPORTED AiirStringRef
aiirDynamicTypeDefinitionGetName(AiirDynamicTypeDefinition typeDef);

/// Get the dialect that the given dynamic type definition belongs to.
AIIR_CAPI_EXPORTED AiirDialect
aiirDynamicTypeDefinitionGetDialect(AiirDynamicTypeDefinition typeDef);

/// Look up a registered attribute definition by attribute name in the given
/// dialect. Note that the dialect must be an extensible dialect.
AIIR_CAPI_EXPORTED AiirDynamicAttrDefinition
aiirExtensibleDialectLookupAttrDefinition(AiirDialect dialect,
                                          AiirStringRef attrName);

/// Check if the given attribute is a dynamic attribute.
AIIR_CAPI_EXPORTED bool aiirAttributeIsADynamicAttr(AiirAttribute attr);

/// Get a dynamic attribute by instantiating the given attribute definition with
/// the provided attributes.
AIIR_CAPI_EXPORTED AiirAttribute aiirDynamicAttrGet(
    AiirDynamicAttrDefinition attrDef, AiirAttribute *attrs, intptr_t numAttrs);

/// Get the number of parameters in the given dynamic attribute.
AIIR_CAPI_EXPORTED intptr_t aiirDynamicAttrGetNumParams(AiirAttribute attr);

/// Get the parameter at the given index in the provided dynamic attribute.
AIIR_CAPI_EXPORTED AiirAttribute aiirDynamicAttrGetParam(AiirAttribute attr,
                                                         intptr_t index);

/// Get the attribute definition of the given dynamic attribute.
AIIR_CAPI_EXPORTED AiirDynamicAttrDefinition
aiirDynamicAttrGetAttrDef(AiirAttribute attr);

/// Get the type ID of a dynamic attribute definition.
AIIR_CAPI_EXPORTED AiirTypeID
aiirDynamicAttrDefinitionGetTypeID(AiirDynamicAttrDefinition attrDef);

/// Get the name of the given dynamic attribute definition.
AIIR_CAPI_EXPORTED AiirStringRef
aiirDynamicAttrDefinitionGetName(AiirDynamicAttrDefinition attrDef);

/// Get the dialect that the given dynamic attribute definition belongs to.
AIIR_CAPI_EXPORTED AiirDialect
aiirDynamicAttrDefinitionGetDialect(AiirDynamicAttrDefinition attrDef);

#ifdef __cplusplus
}
#endif

#endif // AIIR_C_EXTENSIBLEDIALECT_H
