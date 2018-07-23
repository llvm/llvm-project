//===--- IndexDataStoreUtils.cpp - Functions/constants for the data store -===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/Index/IndexDataStoreSymbolUtils.h"
#include "IndexDataStoreUtils.h"
#include "llvm/Bitcode/BitstreamWriter.h"

using namespace clang;
using namespace clang::index;
using namespace clang::index::store;
using namespace llvm;

void store::emitBlockID(unsigned ID, const char *Name,
                        BitstreamWriter &Stream, RecordDataImpl &Record) {
  Record.clear();
  Record.push_back(ID);
  Stream.EmitRecord(bitc::BLOCKINFO_CODE_SETBID, Record);

  // Emit the block name if present.
  if (!Name || Name[0] == 0)
    return;
  Record.clear();
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(bitc::BLOCKINFO_CODE_BLOCKNAME, Record);
}

void store::emitRecordID(unsigned ID, const char *Name,
                         BitstreamWriter &Stream,
                         RecordDataImpl &Record) {
  Record.clear();
  Record.push_back(ID);
  while (*Name)
    Record.push_back(*Name++);
  Stream.EmitRecord(bitc::BLOCKINFO_CODE_SETRECORDNAME, Record);
}

/// Map an indexstore_symbol_kind_t to a SymbolKind, handling unknown values.
SymbolKind index::getSymbolKind(indexstore_symbol_kind_t K) {
  switch ((uint64_t)K) {
  default:
  case INDEXSTORE_SYMBOL_KIND_UNKNOWN:
    return SymbolKind::Unknown;
  case INDEXSTORE_SYMBOL_KIND_MODULE:
    return SymbolKind::Module;
  case INDEXSTORE_SYMBOL_KIND_NAMESPACE:
    return SymbolKind::Namespace;
  case INDEXSTORE_SYMBOL_KIND_NAMESPACEALIAS:
    return SymbolKind::NamespaceAlias;
  case INDEXSTORE_SYMBOL_KIND_MACRO:
    return SymbolKind::Macro;
  case INDEXSTORE_SYMBOL_KIND_ENUM:
    return SymbolKind::Enum;
  case INDEXSTORE_SYMBOL_KIND_STRUCT:
    return SymbolKind::Struct;
  case INDEXSTORE_SYMBOL_KIND_CLASS:
    return SymbolKind::Class;
  case INDEXSTORE_SYMBOL_KIND_PROTOCOL:
    return SymbolKind::Protocol;
  case INDEXSTORE_SYMBOL_KIND_EXTENSION:
    return SymbolKind::Extension;
  case INDEXSTORE_SYMBOL_KIND_UNION:
    return SymbolKind::Union;
  case INDEXSTORE_SYMBOL_KIND_TYPEALIAS:
    return SymbolKind::TypeAlias;
  case INDEXSTORE_SYMBOL_KIND_FUNCTION:
    return SymbolKind::Function;
  case INDEXSTORE_SYMBOL_KIND_VARIABLE:
    return SymbolKind::Variable;
  case INDEXSTORE_SYMBOL_KIND_FIELD:
    return SymbolKind::Field;
  case INDEXSTORE_SYMBOL_KIND_ENUMCONSTANT:
    return SymbolKind::EnumConstant;
  case INDEXSTORE_SYMBOL_KIND_INSTANCEMETHOD:
    return SymbolKind::InstanceMethod;
  case INDEXSTORE_SYMBOL_KIND_CLASSMETHOD:
    return SymbolKind::ClassMethod;
  case INDEXSTORE_SYMBOL_KIND_STATICMETHOD:
    return SymbolKind::StaticMethod;
  case INDEXSTORE_SYMBOL_KIND_INSTANCEPROPERTY:
    return SymbolKind::InstanceProperty;
  case INDEXSTORE_SYMBOL_KIND_CLASSPROPERTY:
    return SymbolKind::ClassProperty;
  case INDEXSTORE_SYMBOL_KIND_STATICPROPERTY:
    return SymbolKind::StaticProperty;
  case INDEXSTORE_SYMBOL_KIND_CONSTRUCTOR:
    return SymbolKind::Constructor;
  case INDEXSTORE_SYMBOL_KIND_DESTRUCTOR:
    return SymbolKind::Destructor;
  case INDEXSTORE_SYMBOL_KIND_CONVERSIONFUNCTION:
    return SymbolKind::ConversionFunction;
  case INDEXSTORE_SYMBOL_KIND_PARAMETER:
    return SymbolKind::Parameter;
  case INDEXSTORE_SYMBOL_KIND_USING:
    return SymbolKind::Using;
  case INDEXSTORE_SYMBOL_KIND_COMMENTTAG:
    return SymbolKind::CommentTag;
  }
}

SymbolSubKind index::getSymbolSubKind(indexstore_symbol_subkind_t K) {
  switch ((uint64_t)K) {
  default:
  case INDEXSTORE_SYMBOL_SUBKIND_NONE:
    return SymbolSubKind::None;
  case INDEXSTORE_SYMBOL_SUBKIND_CXXCOPYCONSTRUCTOR:
    return SymbolSubKind::CXXCopyConstructor;
  case INDEXSTORE_SYMBOL_SUBKIND_CXXMOVECONSTRUCTOR:
    return SymbolSubKind::CXXMoveConstructor;
  case INDEXSTORE_SYMBOL_SUBKIND_ACCESSORGETTER:
      return SymbolSubKind::AccessorGetter;
  case INDEXSTORE_SYMBOL_SUBKIND_ACCESSORSETTER:
      return SymbolSubKind::AccessorSetter;
  case INDEXSTORE_SYMBOL_SUBKIND_USINGTYPENAME:
      return SymbolSubKind::UsingTypename;
  case INDEXSTORE_SYMBOL_SUBKIND_USINGVALUE:
      return SymbolSubKind::UsingValue;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORWILLSET:
    return SymbolSubKind::SwiftAccessorWillSet;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORDIDSET:
    return SymbolSubKind::SwiftAccessorDidSet;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORADDRESSOR:
    return SymbolSubKind::SwiftAccessorAddressor;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORMUTABLEADDRESSOR:
    return SymbolSubKind::SwiftAccessorMutableAddressor;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORREAD:
    return SymbolSubKind::SwiftAccessorRead;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORMODIFY:
    return SymbolSubKind::SwiftAccessorModify;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFSTRUCT:
    return SymbolSubKind::SwiftExtensionOfStruct;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFCLASS:
    return SymbolSubKind::SwiftExtensionOfClass;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFENUM:
    return SymbolSubKind::SwiftExtensionOfEnum;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFPROTOCOL:
    return SymbolSubKind::SwiftExtensionOfProtocol;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTPREFIXOPERATOR:
    return SymbolSubKind::SwiftPrefixOperator;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTPOSTFIXOPERATOR:
    return SymbolSubKind::SwiftPostfixOperator;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTINFIXOPERATOR:
    return SymbolSubKind::SwiftInfixOperator;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTSUBSCRIPT:
    return SymbolSubKind::SwiftSubscript;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTASSOCIATEDTYPE:
    return SymbolSubKind::SwiftAssociatedType;
  case INDEXSTORE_SYMBOL_SUBKIND_SWIFTGENERICTYPEPARAM:
    return SymbolSubKind::SwiftGenericTypeParam;
  }
}

/// Map an indexstore_symbol_language_t to a SymbolLanguage, handling unknown
/// values.
SymbolLanguage index::getSymbolLanguage(indexstore_symbol_language_t L) {
  switch ((uint64_t)L) {
  default: // FIXME: add an unknown language?
  case INDEXSTORE_SYMBOL_LANG_C:
    return SymbolLanguage::C;
  case INDEXSTORE_SYMBOL_LANG_OBJC:
    return SymbolLanguage::ObjC;
  case INDEXSTORE_SYMBOL_LANG_CXX:
    return SymbolLanguage::CXX;
  case INDEXSTORE_SYMBOL_LANG_SWIFT:
    return SymbolLanguage::Swift;
  }
}

/// Map an indexstore representation to a SymbolPropertySet, handling
/// unknown values.
SymbolPropertySet index::getSymbolProperties(uint64_t Props) {
  // FIXME: currently these enums must be kept in sync.
  return (uint64_t)Props;
}

/// Map an indexstore representation to a SymbolRoleSet, handling unknown
/// values.
SymbolRoleSet index::getSymbolRoles(uint64_t Roles) {
  // FIXME: currently these enums must be kept in sync.
  return (uint64_t)Roles;
}

/// Map a SymbolLanguage to a indexstore_symbol_language_t.
indexstore_symbol_kind_t index::getIndexStoreKind(SymbolKind K) {
  switch (K) {
  case SymbolKind::Unknown:
    return INDEXSTORE_SYMBOL_KIND_UNKNOWN;
  case SymbolKind::Module:
    return INDEXSTORE_SYMBOL_KIND_MODULE;
  case SymbolKind::Namespace:
    return INDEXSTORE_SYMBOL_KIND_NAMESPACE;
  case SymbolKind::NamespaceAlias:
    return INDEXSTORE_SYMBOL_KIND_NAMESPACEALIAS;
  case SymbolKind::Macro:
    return INDEXSTORE_SYMBOL_KIND_MACRO;
  case SymbolKind::Enum:
    return INDEXSTORE_SYMBOL_KIND_ENUM;
  case SymbolKind::Struct:
    return INDEXSTORE_SYMBOL_KIND_STRUCT;
  case SymbolKind::Class:
    return INDEXSTORE_SYMBOL_KIND_CLASS;
  case SymbolKind::Protocol:
    return INDEXSTORE_SYMBOL_KIND_PROTOCOL;
  case SymbolKind::Extension:
    return INDEXSTORE_SYMBOL_KIND_EXTENSION;
  case SymbolKind::Union:
    return INDEXSTORE_SYMBOL_KIND_UNION;
  case SymbolKind::TypeAlias:
    return INDEXSTORE_SYMBOL_KIND_TYPEALIAS;
  case SymbolKind::Function:
    return INDEXSTORE_SYMBOL_KIND_FUNCTION;
  case SymbolKind::Variable:
    return INDEXSTORE_SYMBOL_KIND_VARIABLE;
  case SymbolKind::Field:
    return INDEXSTORE_SYMBOL_KIND_FIELD;
  case SymbolKind::EnumConstant:
    return INDEXSTORE_SYMBOL_KIND_ENUMCONSTANT;
  case SymbolKind::InstanceMethod:
    return INDEXSTORE_SYMBOL_KIND_INSTANCEMETHOD;
  case SymbolKind::ClassMethod:
    return INDEXSTORE_SYMBOL_KIND_CLASSMETHOD;
  case SymbolKind::StaticMethod:
    return INDEXSTORE_SYMBOL_KIND_STATICMETHOD;
  case SymbolKind::InstanceProperty:
    return INDEXSTORE_SYMBOL_KIND_INSTANCEPROPERTY;
  case SymbolKind::ClassProperty:
    return INDEXSTORE_SYMBOL_KIND_CLASSPROPERTY;
  case SymbolKind::StaticProperty:
    return INDEXSTORE_SYMBOL_KIND_STATICPROPERTY;
  case SymbolKind::Constructor:
    return INDEXSTORE_SYMBOL_KIND_CONSTRUCTOR;
  case SymbolKind::Destructor:
    return INDEXSTORE_SYMBOL_KIND_DESTRUCTOR;
  case SymbolKind::ConversionFunction:
    return INDEXSTORE_SYMBOL_KIND_CONVERSIONFUNCTION;
  case SymbolKind::Parameter:
    return INDEXSTORE_SYMBOL_KIND_PARAMETER;
  case SymbolKind::Using:
    return INDEXSTORE_SYMBOL_KIND_USING;
  case SymbolKind::CommentTag:
    return INDEXSTORE_SYMBOL_KIND_COMMENTTAG;
  }
  llvm_unreachable("unexpected symbol kind");
}

indexstore_symbol_subkind_t index::getIndexStoreSubKind(SymbolSubKind K) {
  switch (K) {
  case SymbolSubKind::None:
    return INDEXSTORE_SYMBOL_SUBKIND_NONE;
  case SymbolSubKind::CXXCopyConstructor:
    return INDEXSTORE_SYMBOL_SUBKIND_CXXCOPYCONSTRUCTOR;
  case SymbolSubKind::CXXMoveConstructor:
    return INDEXSTORE_SYMBOL_SUBKIND_CXXMOVECONSTRUCTOR;
  case SymbolSubKind::AccessorGetter:
    return INDEXSTORE_SYMBOL_SUBKIND_ACCESSORGETTER;
  case SymbolSubKind::AccessorSetter:
    return INDEXSTORE_SYMBOL_SUBKIND_ACCESSORSETTER;
  case SymbolSubKind::UsingTypename:
    return INDEXSTORE_SYMBOL_SUBKIND_USINGTYPENAME;
  case SymbolSubKind::UsingValue:
    return INDEXSTORE_SYMBOL_SUBKIND_USINGVALUE;
  case SymbolSubKind::SwiftAccessorWillSet:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORWILLSET;
  case SymbolSubKind::SwiftAccessorDidSet:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORDIDSET;
  case SymbolSubKind::SwiftAccessorAddressor:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORADDRESSOR;
  case SymbolSubKind::SwiftAccessorMutableAddressor:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORMUTABLEADDRESSOR;
  case SymbolSubKind::SwiftAccessorRead:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORREAD;
  case SymbolSubKind::SwiftAccessorModify:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTACCESSORMODIFY;
  case SymbolSubKind::SwiftExtensionOfStruct:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFSTRUCT;
  case SymbolSubKind::SwiftExtensionOfClass:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFCLASS;
  case SymbolSubKind::SwiftExtensionOfEnum:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFENUM;
  case SymbolSubKind::SwiftExtensionOfProtocol:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTEXTENSIONOFPROTOCOL;
  case SymbolSubKind::SwiftPrefixOperator:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTPREFIXOPERATOR;
  case SymbolSubKind::SwiftPostfixOperator:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTPOSTFIXOPERATOR;
  case SymbolSubKind::SwiftInfixOperator:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTINFIXOPERATOR;
  case SymbolSubKind::SwiftSubscript:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTSUBSCRIPT;
  case SymbolSubKind::SwiftAssociatedType:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTASSOCIATEDTYPE;
  case SymbolSubKind::SwiftGenericTypeParam:
    return INDEXSTORE_SYMBOL_SUBKIND_SWIFTGENERICTYPEPARAM;
  }
  llvm_unreachable("unexpected symbol subkind");
}

/// Map a SymbolLanguage to a indexstore_symbol_language_t.
indexstore_symbol_language_t index::getIndexStoreLang(SymbolLanguage L) {
  switch (L) {
  case SymbolLanguage::C:
    return INDEXSTORE_SYMBOL_LANG_C;
  case SymbolLanguage::ObjC:
    return INDEXSTORE_SYMBOL_LANG_OBJC;
  case SymbolLanguage::CXX:
    return INDEXSTORE_SYMBOL_LANG_CXX;
  case SymbolLanguage::Swift:
    return INDEXSTORE_SYMBOL_LANG_SWIFT;
  }
  llvm_unreachable("unexpected symbol language");
}

/// Map a SymbolPropertySet to its indexstore representation.
uint64_t index::getIndexStoreProperties(SymbolPropertySet Props) {
  uint64_t storeProp = 0;
  applyForEachSymbolProperty(Props, [&](SymbolProperty prop) {
    switch (prop) {
    case SymbolProperty::Generic:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_GENERIC;
      break;
    case SymbolProperty::TemplatePartialSpecialization:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_TEMPLATE_PARTIAL_SPECIALIZATION;
      break;
    case SymbolProperty::TemplateSpecialization:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_TEMPLATE_SPECIALIZATION;
      break;
    case SymbolProperty::UnitTest:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_UNITTEST;
      break;
    case SymbolProperty::IBAnnotated:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_IBANNOTATED;
      break;
    case SymbolProperty::IBOutletCollection:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_IBOUTLETCOLLECTION;
      break;
    case SymbolProperty::GKInspectable:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_GKINSPECTABLE;
      break;
    case SymbolProperty::Local:
      storeProp |= INDEXSTORE_SYMBOL_PROPERTY_LOCAL;
      break;
    }
  });
  return storeProp;
}

/// Map a SymbolRoleSet to its indexstore representation.
uint64_t index::getIndexStoreRoles(SymbolRoleSet Roles) {
  uint64_t storeRoles = 0;
  applyForEachSymbolRole(Roles, [&](SymbolRole role) {
    switch (role) {
    case SymbolRole::Declaration:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_DECLARATION;
      break;
    case SymbolRole::Definition:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_DEFINITION;
      break;
    case SymbolRole::Reference:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REFERENCE;
      break;
    case SymbolRole::Read:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_READ;
      break;
    case SymbolRole::Write:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_WRITE;
      break;
    case SymbolRole::Call:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_CALL;
      break;
    case SymbolRole::Dynamic:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_DYNAMIC;
      break;
    case SymbolRole::AddressOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_ADDRESSOF;
      break;
    case SymbolRole::Implicit:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_IMPLICIT;
      break;
    case SymbolRole::RelationChildOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_CHILDOF;
      break;
    case SymbolRole::RelationBaseOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_BASEOF;
      break;
    case SymbolRole::RelationOverrideOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_OVERRIDEOF;
      break;
    case SymbolRole::RelationReceivedBy:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_RECEIVEDBY;
      break;
    case SymbolRole::RelationCalledBy:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_CALLEDBY;
      break;
    case SymbolRole::RelationExtendedBy:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_EXTENDEDBY;
      break;
    case SymbolRole::RelationAccessorOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_ACCESSOROF;
      break;
    case SymbolRole::RelationContainedBy:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_CONTAINEDBY;
      break;
    case SymbolRole::RelationIBTypeOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_IBTYPEOF;
      break;
    case SymbolRole::RelationSpecializationOf:
      storeRoles |= INDEXSTORE_SYMBOL_ROLE_REL_SPECIALIZATIONOF;
      break;
    }
  });
  return storeRoles;
}
