//==--- AbstractBasicWriter.h - Abstract basic value serialization --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_AST_ABSTRACTBASICWRITER_H
#define CLANG_AST_ABSTRACTBASICWRITER_H

#include "clang/AST/DeclTemplate.h"

namespace clang {
namespace serialization {

template <class T>
inline llvm::Optional<T> makeOptionalFromNullable(const T &value) {
  return (value.isNull()
            ? llvm::Optional<T>()
            : llvm::Optional<T>(value));
}

template <class T>
inline llvm::Optional<T*> makeOptionalFromPointer(T *value) {
  return (value ? llvm::Optional<T*>(value) : llvm::Optional<T*>());
}

// PropertyWriter is a class concept that requires the following method:
//   BasicWriter find(llvm::StringRef propertyName);
// where BasicWriter is some class conforming to the BasicWriter concept.
// An abstract AST-node writer is created with a PropertyWriter and
// performs a sequence of calls like so:
//   propertyWriter.find(propertyName).write##TypeName(value)
// to write the properties of the node it is serializing.

// BasicWriter is a class concept that requires methods like:
//   void write##TypeName(ValueType value);
// where TypeName is the name of a PropertyType node from PropertiesBase.td
// and ValueType is the corresponding C++ type name.

// WriteDispatcher is a template which does type-based forwarding to one
// of the write methods of the BasicWriter passed in:
//
// template <class ValueType>
// struct WriteDispatcher {
//   template <class BasicWriter>
//   static void write(BasicWriter &W, ValueType value);
// };

// BasicWriterBase provides convenience implementations of the write
// methods for EnumPropertyType and SubclassPropertyType types that just
// defer to the "underlying" implementations (for UInt32 and the base class,
// respectively).
//
// template <class Impl>
// class BasicWriterBase {
// protected:
//   Impl &asImpl();
// public:
//   ...
// };

// The actual classes are auto-generated; see ClangASTPropertiesEmitter.cpp.
#include "clang/AST/AbstractBasicWriter.inc"

/// DataStreamBasicWriter provides convenience implementations for many
/// BasicWriter methods based on the assumption that the
/// ultimate writer implementation is based on a variable-length stream
/// of unstructured data (like Clang's module files).  It is designed
/// to pair with DataStreamBasicReader.
///
/// This class can also act as a PropertyWriter, implementing find("...")
/// by simply forwarding to itself.
///
/// Unimplemented methods:
///   writeBool
///   writeUInt32
///   writeUInt64
///   writeIdentifier
///   writeSelector
///   writeSourceLocation
///   writeQualType
///   writeStmtRef
///   writeDeclRef
template <class Impl>
class DataStreamBasicWriter : public BasicWriterBase<Impl> {
protected:
  using BasicWriterBase<Impl>::asImpl;

public:
  /// Implement property-find by ignoring it.  We rely on properties being
  /// serialized and deserialized in a reliable order instead.
  Impl &find(const char *propertyName) {
    return asImpl();
  }

  template <class T>
  void writeArray(llvm::ArrayRef<T> array) {
    asImpl().writeUInt32(array.size());
    for (const T &elt : array) {
      WriteDispatcher<T>::write(asImpl(), elt);
    }
  }

  template <class T>
  void writeOptional(llvm::Optional<T> value) {
    WriteDispatcher<T>::write(asImpl(), PackOptionalValue<T>::pack(value));
  }

  void writeAPSInt(const llvm::APSInt &value) {
    asImpl().writeBool(value.isUnsigned());
    asImpl().writeAPInt(value);
  }

  void writeAPInt(const llvm::APInt &value) {
    asImpl().writeUInt32(value.getBitWidth());
    const uint64_t *words = value.getRawData();
    for (size_t i = 0, e = value.getNumWords(); i != e; ++i)
      asImpl().writeUInt64(words[i]);
  }

  void writeQualifiers(Qualifiers value) {
    static_assert(sizeof(value.getAsOpaqueValue()) <= sizeof(uint32_t),
                  "update this if the value size changes");
    asImpl().writeUInt32(value.getAsOpaqueValue());
  }

  void writeExceptionSpecInfo(
                        const FunctionProtoType::ExceptionSpecInfo &esi) {
    asImpl().writeUInt32(uint32_t(esi.Type));
    if (esi.Type == EST_Dynamic) {
      asImpl().writeArray(esi.Exceptions);
    } else if (isComputedNoexcept(esi.Type)) {
      asImpl().writeExprRef(esi.NoexceptExpr);
    } else if (esi.Type == EST_Uninstantiated) {
      asImpl().writeDeclRef(esi.SourceDecl);
      asImpl().writeDeclRef(esi.SourceTemplate);
    } else if (esi.Type == EST_Unevaluated) {
      asImpl().writeDeclRef(esi.SourceDecl);
    }
  }

  void writeExtParameterInfo(FunctionProtoType::ExtParameterInfo epi) {
    static_assert(sizeof(epi.getOpaqueValue()) <= sizeof(uint32_t),
                  "opaque value doesn't fit into uint32_t");
    asImpl().writeUInt32(epi.getOpaqueValue());
  }

  void writeDeclarationName(DeclarationName name) {
    asImpl().writeDeclarationNameKind(name.getNameKind());
    switch (name.getNameKind()) {
    case DeclarationName::Identifier:
      asImpl().writeIdentifier(name.getAsIdentifierInfo());
      return;

    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      asImpl().writeSelector(name.getObjCSelector());
      return;

    case DeclarationName::CXXConstructorName:
    case DeclarationName::CXXDestructorName:
    case DeclarationName::CXXConversionFunctionName:
      asImpl().writeQualType(name.getCXXNameType());
      return;

    case DeclarationName::CXXDeductionGuideName:
      asImpl().writeDeclRef(name.getCXXDeductionGuideTemplate());
      return;

    case DeclarationName::CXXOperatorName:
      asImpl().writeOverloadedOperatorKind(name.getCXXOverloadedOperator());
      return;

    case DeclarationName::CXXLiteralOperatorName:
      asImpl().writeIdentifier(name.getCXXLiteralIdentifier());
      return;

    case DeclarationName::CXXUsingDirective:
      // No extra data to emit
      return;
    }
    llvm_unreachable("bad name kind");
  }

  void writeTemplateName(TemplateName name) {
    asImpl().writeTemplateNameKind(name.getKind());
    switch (name.getKind()) {
    case TemplateName::Template:
      asImpl().writeDeclRef(name.getAsTemplateDecl());
      return;

    case TemplateName::OverloadedTemplate: {
      OverloadedTemplateStorage *overload = name.getAsOverloadedTemplate();
      asImpl().writeArray(llvm::makeArrayRef(overload->begin(),
                                             overload->end()));
      return;
    }

    case TemplateName::AssumedTemplate: {
      AssumedTemplateStorage *assumed = name.getAsAssumedTemplateName();
      asImpl().writeDeclarationName(assumed->getDeclName());
      return;
    }

    case TemplateName::QualifiedTemplate: {
      QualifiedTemplateName *qual = name.getAsQualifiedTemplateName();
      asImpl().writeNestedNameSpecifier(qual->getQualifier());
      asImpl().writeBool(qual->hasTemplateKeyword());
      asImpl().writeDeclRef(qual->getTemplateDecl());
      return;
    }

    case TemplateName::DependentTemplate: {
      DependentTemplateName *dep = name.getAsDependentTemplateName();
      asImpl().writeNestedNameSpecifier(dep->getQualifier());
      asImpl().writeBool(dep->isIdentifier());
      if (dep->isIdentifier())
        asImpl().writeIdentifier(dep->getIdentifier());
      else
        asImpl().writeOverloadedOperatorKind(dep->getOperator());
      return;
    }

    case TemplateName::SubstTemplateTemplateParm: {
      auto subst = name.getAsSubstTemplateTemplateParm();
      asImpl().writeDeclRef(subst->getParameter());
      asImpl().writeTemplateName(subst->getReplacement());
      return;
    }

    case TemplateName::SubstTemplateTemplateParmPack: {
      auto substPack = name.getAsSubstTemplateTemplateParmPack();
      asImpl().writeDeclRef(substPack->getParameterPack());
      asImpl().writeTemplateArgument(substPack->getArgumentPack());
      return;
    }
    }
    llvm_unreachable("bad template name kind");
  }

  void writeTemplateArgument(const TemplateArgument &arg) {
    asImpl().writeTemplateArgumentKind(arg.getKind());
    switch (arg.getKind()) {
    case TemplateArgument::Null:
      return;
    case TemplateArgument::Type:
      asImpl().writeQualType(arg.getAsType());
      return;
    case TemplateArgument::Declaration:
      asImpl().writeValueDeclRef(arg.getAsDecl());
      asImpl().writeQualType(arg.getParamTypeForDecl());
      return;
    case TemplateArgument::NullPtr:
      asImpl().writeQualType(arg.getNullPtrType());
      return;
    case TemplateArgument::Integral:
      asImpl().writeAPSInt(arg.getAsIntegral());
      asImpl().writeQualType(arg.getIntegralType());
      return;
    case TemplateArgument::Template:
      asImpl().writeTemplateName(arg.getAsTemplateOrTemplatePattern());
      return;
    case TemplateArgument::TemplateExpansion: {
      asImpl().writeTemplateName(arg.getAsTemplateOrTemplatePattern());
      // Convert Optional<unsigned> to Optional<uint32>, just in case.
      Optional<unsigned> numExpansions = arg.getNumTemplateExpansions();
      Optional<uint32_t> numExpansions32;
      if (numExpansions) numExpansions32 = *numExpansions;
      asImpl().template writeOptional<uint32_t>(numExpansions32);
      return;
    }
    case TemplateArgument::Expression:
      asImpl().writeExprRef(arg.getAsExpr());
      return;
    case TemplateArgument::Pack:
      asImpl().template writeArray<TemplateArgument>(arg.pack_elements());
      return;
    }
    llvm_unreachable("bad template argument kind");
  }

  void writeNestedNameSpecifier(NestedNameSpecifier *NNS) {
    // Nested name specifiers usually aren't too long. I think that 8 would
    // typically accommodate the vast majority.
    SmallVector<NestedNameSpecifier *, 8> nestedNames;

    // Push each of the NNS's onto a stack for serialization in reverse order.
    while (NNS) {
      nestedNames.push_back(NNS);
      NNS = NNS->getPrefix();
    }

    asImpl().writeUInt32(nestedNames.size());
    while (!nestedNames.empty()) {
      NNS = nestedNames.pop_back_val();
      NestedNameSpecifier::SpecifierKind kind = NNS->getKind();
      asImpl().writeNestedNameSpecifierKind(kind);
      switch (kind) {
      case NestedNameSpecifier::Identifier:
        asImpl().writeIdentifier(NNS->getAsIdentifier());
        continue;

      case NestedNameSpecifier::Namespace:
        asImpl().writeNamespaceDeclRef(NNS->getAsNamespace());
        continue;

      case NestedNameSpecifier::NamespaceAlias:
        asImpl().writeNamespaceAliasDeclRef(NNS->getAsNamespaceAlias());
        continue;

      case NestedNameSpecifier::TypeSpec:
      case NestedNameSpecifier::TypeSpecWithTemplate:
        asImpl().writeQualType(QualType(NNS->getAsType(), 0));
        continue;

      case NestedNameSpecifier::Global:
        // Don't need to write an associated value.
        continue;

      case NestedNameSpecifier::Super:
        asImpl().writeDeclRef(NNS->getAsRecordDecl());
        continue;
      }
      llvm_unreachable("bad nested name specifier kind");
    }
  }
};

} // end namespace serialization
} // end namespace clang

#endif
