//===- Property.cpp - Property wrapper class ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Property wrapper to simplify using TableGen Record defining a AIIR
// Property.
//
//===----------------------------------------------------------------------===//

#include "aiir/TableGen/Property.h"
#include "aiir/TableGen/Operator.h"
#include "aiir/TableGen/Predicate.h"
#include "llvm/TableGen/Record.h"

using namespace aiir;
using namespace aiir::tblgen;

using llvm::DefInit;
using llvm::Init;
using llvm::Record;
using llvm::StringInit;

// Returns the initializer's value as string if the given TableGen initializer
// is a code or string initializer. Returns the empty StringRef otherwise.
static StringRef getValueAsString(const Init *init) {
  if (const auto *str = dyn_cast<StringInit>(init))
    return str->getValue().trim();
  return {};
}

StringRef PropConstraint::getInterfaceType() const {
  return getValueAsString(def->getValueInit("interfaceType"));
}

Property::Property(const Record *def)
    : Property(
          def, getValueAsString(def->getValueInit("summary")),
          getValueAsString(def->getValueInit("description")),
          getValueAsString(def->getValueInit("storageType")),
          getValueAsString(def->getValueInit("interfaceType")),
          getValueAsString(def->getValueInit("convertFromStorage")),
          getValueAsString(def->getValueInit("assignToStorage")),
          getValueAsString(def->getValueInit("convertToAttribute")),
          getValueAsString(def->getValueInit("convertFromAttribute")),
          getValueAsString(def->getValueInit("parser")),
          getValueAsString(def->getValueInit("optionalParser")),
          getValueAsString(def->getValueInit("printer")),
          getValueAsString(def->getValueInit("readFromAiirBytecode")),
          getValueAsString(def->getValueInit("writeToAiirBytecode")),
          getValueAsString(def->getValueInit("hashProperty")),
          getValueAsString(def->getValueInit("defaultValue")),
          getValueAsString(def->getValueInit("storageTypeValueOverride"))) {
  assert((def->isSubClassOf("Property") || def->isSubClassOf("Attr")) &&
         "must be subclass of TableGen 'Property' class");
}

Property::Property(const DefInit *init) : Property(init->getDef()) {}

Property::Property(const llvm::Record *maybeDef, StringRef summary,
                   StringRef description, StringRef storageType,
                   StringRef interfaceType, StringRef convertFromStorageCall,
                   StringRef assignToStorageCall,
                   StringRef convertToAttributeCall,
                   StringRef convertFromAttributeCall, StringRef parserCall,
                   StringRef optionalParserCall, StringRef printerCall,
                   StringRef readFromAiirBytecodeCall,
                   StringRef writeToAiirBytecodeCall,
                   StringRef hashPropertyCall, StringRef defaultValue,
                   StringRef storageTypeValueOverride)
    : PropConstraint(maybeDef, Constraint::CK_Prop), summary(summary),
      description(description), storageType(storageType),
      interfaceType(interfaceType),
      convertFromStorageCall(convertFromStorageCall),
      assignToStorageCall(assignToStorageCall),
      convertToAttributeCall(convertToAttributeCall),
      convertFromAttributeCall(convertFromAttributeCall),
      parserCall(parserCall), optionalParserCall(optionalParserCall),
      printerCall(printerCall),
      readFromAiirBytecodeCall(readFromAiirBytecodeCall),
      writeToAiirBytecodeCall(writeToAiirBytecodeCall),
      hashPropertyCall(hashPropertyCall), defaultValue(defaultValue),
      storageTypeValueOverride(storageTypeValueOverride) {
  if (storageType.empty())
    storageType = "Property";
}

StringRef Property::getPropertyDefName() const {
  if (def->isAnonymous()) {
    return getBaseProperty().def->getName();
  }
  return def->getName();
}

Pred Property::getPredicate() const {
  if (!def)
    return Pred();
  const llvm::RecordVal *maybePred = def->getValue("predicate");
  if (!maybePred || !maybePred->getValue())
    return Pred();
  return Pred(maybePred->getValue());
}

Property Property::getBaseProperty() const {
  if (const auto *defInit =
          llvm::dyn_cast<llvm::DefInit>(def->getValueInit("baseProperty"))) {
    return Property(defInit).getBaseProperty();
  }
  return *this;
}

bool Property::isSubClassOf(StringRef className) const {
  return def && def->isSubClassOf(className);
}

StringRef ConstantProp::getValue() const {
  return def->getValueAsString("value");
}
