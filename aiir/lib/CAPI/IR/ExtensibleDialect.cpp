//===- ExtensibleDialect - C API for AIIR Extensible Dialect --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aiir-c/ExtensibleDialect.h"
#include "aiir/CAPI/IR.h"
#include "aiir/CAPI/Support.h"
#include "aiir/IR/ExtensibleDialect.h"
#include "aiir/IR/OperationSupport.h"

using namespace aiir;

DEFINE_C_API_PTR_METHODS(AiirDynamicOpTrait, DynamicOpTrait)
DEFINE_C_API_PTR_METHODS(AiirDynamicTypeDefinition, DynamicTypeDefinition)
DEFINE_C_API_PTR_METHODS(AiirDynamicAttrDefinition, DynamicAttrDefinition)

bool aiirDynamicOpTraitAttach(AiirDynamicOpTrait dynamicOpTrait,
                              AiirStringRef opName, AiirContext context) {
  std::optional<RegisteredOperationName> opNameFound =
      RegisteredOperationName::lookup(unwrap(opName), unwrap(context));
  assert(opNameFound && "operation name must be registered in the context");

  // The original getImpl() is protected, so we create a small helper struct
  // here.
  struct RegisteredOperationNameWithImpl : RegisteredOperationName {
    Impl *getImpl() { return RegisteredOperationName::getImpl(); }
  };
  OperationName::Impl *impl =
      static_cast<RegisteredOperationNameWithImpl &>(*opNameFound).getImpl();

  std::unique_ptr<DynamicOpTrait> trait(unwrap(dynamicOpTrait));
  // TODO: we should enable llvm-style RTTI for `OperationName::Impl` and check
  // whether the `impl` is a `DynamicOpDefinition` here.
  return static_cast<DynamicOpDefinition *>(impl)->addTrait(std::move(trait));
}

AiirDynamicOpTrait aiirDynamicOpTraitIsTerminatorCreate() {
  return wrap(new DynamicOpTraits::IsTerminator());
}

AiirTypeID aiirDynamicOpTraitIsTerminatorGetTypeID() {
  return wrap(DynamicOpTraits::IsTerminator::getStaticTypeID());
}

AiirDynamicOpTrait aiirDynamicOpTraitNoTerminatorCreate() {
  return wrap(new DynamicOpTraits::NoTerminator());
}

AiirTypeID aiirDynamicOpTraitNoTerminatorGetTypeID() {
  return wrap(DynamicOpTraits::NoTerminator::getStaticTypeID());
}

void aiirDynamicOpTraitDestroy(AiirDynamicOpTrait dynamicOpTrait) {
  delete unwrap(dynamicOpTrait);
}

namespace aiir {

class ExternalDynamicOpTrait : public DynamicOpTrait {
public:
  ExternalDynamicOpTrait(TypeID typeID, AiirDynamicOpTraitCallbacks callbacks,
                         void *userData)
      : typeID(typeID), callbacks(callbacks), userData(userData) {
    if (callbacks.construct)
      callbacks.construct(userData);
  }
  ~ExternalDynamicOpTrait() {
    if (callbacks.destruct)
      callbacks.destruct(userData);
  }

  LogicalResult verifyTrait(Operation *op) const override {
    return unwrap(callbacks.verifyTrait(wrap(op), userData));
  };
  LogicalResult verifyRegionTrait(Operation *op) const override {
    return unwrap(callbacks.verifyRegionTrait(wrap(op), userData));
  };

  TypeID getTypeID() const override { return typeID; };

private:
  TypeID typeID;
  AiirDynamicOpTraitCallbacks callbacks;
  void *userData;
};

} // namespace aiir

AiirDynamicOpTrait aiirDynamicOpTraitCreate(
    AiirTypeID typeID, AiirDynamicOpTraitCallbacks callbacks, void *userData) {
  return wrap(
      new aiir::ExternalDynamicOpTrait(unwrap(typeID), callbacks, userData));
}

bool aiirDialectIsAExtensibleDialect(AiirDialect dialect) {
  return llvm::isa<aiir::ExtensibleDialect>(unwrap(dialect));
}

AiirDynamicTypeDefinition
aiirExtensibleDialectLookupTypeDefinition(AiirDialect dialect,
                                          AiirStringRef typeName) {
  return wrap(llvm::cast<aiir::ExtensibleDialect>(unwrap(dialect))
                  ->lookupTypeDefinition(unwrap(typeName)));
}

bool aiirTypeIsADynamicType(AiirType type) {
  return llvm::isa<aiir::DynamicType>(unwrap(type));
}

AiirTypeID aiirDynamicTypeGetTypeID() {
  return wrap(aiir::DynamicType::getTypeID());
}

AiirType aiirDynamicTypeGet(AiirDynamicTypeDefinition typeDef,
                            AiirAttribute *attrs, intptr_t numAttrs) {
  llvm::SmallVector<aiir::Attribute> attributes;
  attributes.reserve(numAttrs);
  for (intptr_t i = 0; i < numAttrs; ++i)
    attributes.push_back(unwrap(attrs[i]));

  return wrap(aiir::DynamicType::get(unwrap(typeDef), attributes));
}

intptr_t aiirDynamicTypeGetNumParams(AiirType type) {
  return llvm::cast<aiir::DynamicType>(unwrap(type)).getParams().size();
}

AiirAttribute aiirDynamicTypeGetParam(AiirType type, intptr_t index) {
  return wrap(llvm::cast<aiir::DynamicType>(unwrap(type)).getParams()[index]);
}

AiirDynamicTypeDefinition aiirDynamicTypeGetTypeDef(AiirType type) {
  return wrap(llvm::cast<aiir::DynamicType>(unwrap(type)).getTypeDef());
}

AiirTypeID
aiirDynamicTypeDefinitionGetTypeID(AiirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getTypeID());
}

AiirStringRef
aiirDynamicTypeDefinitionGetName(AiirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getName());
}

AiirDialect
aiirDynamicTypeDefinitionGetDialect(AiirDynamicTypeDefinition typeDef) {
  return wrap(unwrap(typeDef)->getDialect());
}

AiirDynamicAttrDefinition
aiirExtensibleDialectLookupAttrDefinition(AiirDialect dialect,
                                          AiirStringRef attrName) {
  return wrap(llvm::cast<aiir::ExtensibleDialect>(unwrap(dialect))
                  ->lookupAttrDefinition(unwrap(attrName)));
}

bool aiirAttributeIsADynamicAttr(AiirAttribute attr) {
  return llvm::isa<aiir::DynamicAttr>(unwrap(attr));
}

AiirTypeID aiirDynamicAttrGetTypeID(void) {
  return wrap(aiir::DynamicAttr::getTypeID());
}

AiirAttribute aiirDynamicAttrGet(AiirDynamicAttrDefinition attrDef,
                                 AiirAttribute *attrs, intptr_t numAttrs) {
  llvm::SmallVector<aiir::Attribute> attributes;
  attributes.reserve(numAttrs);
  for (intptr_t i = 0; i < numAttrs; ++i)
    attributes.push_back(unwrap(attrs[i]));

  return wrap(aiir::DynamicAttr::get(unwrap(attrDef), attributes));
}

intptr_t aiirDynamicAttrGetNumParams(AiirAttribute attr) {
  return llvm::cast<aiir::DynamicAttr>(unwrap(attr)).getParams().size();
}

AiirAttribute aiirDynamicAttrGetParam(AiirAttribute attr, intptr_t index) {
  return wrap(llvm::cast<aiir::DynamicAttr>(unwrap(attr)).getParams()[index]);
}

AiirDynamicAttrDefinition aiirDynamicAttrGetAttrDef(AiirAttribute attr) {
  return wrap(llvm::cast<aiir::DynamicAttr>(unwrap(attr)).getAttrDef());
}

AiirTypeID
aiirDynamicAttrDefinitionGetTypeID(AiirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getTypeID());
}

AiirStringRef
aiirDynamicAttrDefinitionGetName(AiirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getName());
}

AiirDialect
aiirDynamicAttrDefinitionGetDialect(AiirDynamicAttrDefinition attrDef) {
  return wrap(unwrap(attrDef)->getDialect());
}
