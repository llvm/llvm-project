//==--- AbstractBasiceReader.h - Abstract basic value deserialization -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_AST_ABSTRACTBASICREADER_H
#define CLANG_AST_ABSTRACTBASICREADER_H

#include "clang/AST/DeclTemplate.h"

namespace clang {
namespace serialization {

template <class T>
inline T makeNullableFromOptional(const Optional<T> &value) {
  return (value ? *value : T());
}

template <class T>
inline T *makePointerFromOptional(Optional<T *> value) {
  return (value ? *value : nullptr);
}

// PropertyReader is a class concept that requires the following method:
//   BasicReader find(llvm::StringRef propertyName);
// where BasicReader is some class conforming to the BasicReader concept.
// An abstract AST-node reader is created with a PropertyReader and
// performs a sequence of calls like so:
//   propertyReader.find(propertyName).read##TypeName()
// to read the properties of the node it is deserializing.

// BasicReader is a class concept that requires methods like:
//   ValueType read##TypeName();
// where TypeName is the name of a PropertyType node from PropertiesBase.td
// and ValueType is the corresponding C++ type name.  The read method may
// require one or more buffer arguments.

// ReadDispatcher does type-based forwarding to one of the read methods
// on the BasicReader passed in:
//
// template <class ValueType>
// struct ReadDispatcher {
//   template <class BasicReader, class... BufferTypes>
//   static ValueType read(BasicReader &R, BufferTypes &&...);
// };

// BasicReaderBase provides convenience implementations of the read methods
// for EnumPropertyType and SubclassPropertyType types that just defer to
// the "underlying" implementations (for UInt32 and the base class,
// respectively).
//
// template <class Impl>
// class BasicReaderBase {
// protected:
//   BasicReaderBase(ASTContext &ctx);
//   Impl &asImpl();
// public:
//   ASTContext &getASTContext();
//   ...
// };

// The actual classes are auto-generated; see ClangASTPropertiesEmitter.cpp.
#include "clang/AST/AbstractBasicReader.inc"

/// DataStreamBasicReader provides convenience implementations for many
/// BasicReader methods based on the assumption that the
/// ultimate reader implementation is based on a variable-length stream
/// of unstructured data (like Clang's module files).  It is designed
/// to pair with DataStreamBasicWriter.
///
/// This class can also act as a PropertyReader, implementing find("...")
/// by simply forwarding to itself.
///
/// Unimplemented methods:
///   readBool
///   readUInt32
///   readUInt64
///   readIdentifier
///   readSelector
///   readSourceLocation
///   readQualType
///   readStmtRef
///   readDeclRef
template <class Impl>
class DataStreamBasicReader : public BasicReaderBase<Impl> {
protected:
  using BasicReaderBase<Impl>::asImpl;
  DataStreamBasicReader(ASTContext &ctx) : BasicReaderBase<Impl>(ctx) {}

public:
  using BasicReaderBase<Impl>::getASTContext;

  /// Implement property-find by ignoring it.  We rely on properties being
  /// serialized and deserialized in a reliable order instead.
  Impl &find(const char *propertyName) {
    return asImpl();
  }

  template <class T>
  llvm::ArrayRef<T> readArray(llvm::SmallVectorImpl<T> &buffer) {
    assert(buffer.empty());

    uint32_t size = asImpl().readUInt32();
    buffer.reserve(size);

    for (uint32_t i = 0; i != size; ++i) {
      buffer.push_back(ReadDispatcher<T>::read(asImpl()));
    }
    return buffer;
  }

  template <class T, class... Args>
  llvm::Optional<T> readOptional(Args &&...args) {
    return UnpackOptionalValue<T>::unpack(
             ReadDispatcher<T>::read(asImpl(), std::forward<Args>(args)...));
  }

  llvm::APSInt readAPSInt() {
    bool isUnsigned = asImpl().readBool();
    llvm::APInt value = asImpl().readAPInt();
    return llvm::APSInt(std::move(value), isUnsigned);
  }

  llvm::APInt readAPInt() {
    unsigned bitWidth = asImpl().readUInt32();
    unsigned numWords = llvm::APInt::getNumWords(bitWidth);
    llvm::SmallVector<uint64_t, 4> data;
    for (uint32_t i = 0; i != numWords; ++i)
      data.push_back(asImpl().readUInt64());
    return llvm::APInt(bitWidth, numWords, &data[0]);
  }

  Qualifiers readQualifiers() {
    static_assert(sizeof(Qualifiers().getAsOpaqueValue()) <= sizeof(uint32_t),
                  "update this if the value size changes");
    uint32_t value = asImpl().readUInt32();
    return Qualifiers::fromOpaqueValue(value);
  }

  FunctionProtoType::ExceptionSpecInfo
  readExceptionSpecInfo(llvm::SmallVectorImpl<QualType> &buffer) {
    FunctionProtoType::ExceptionSpecInfo esi;
    esi.Type = ExceptionSpecificationType(asImpl().readUInt32());
    if (esi.Type == EST_Dynamic) {
      esi.Exceptions = asImpl().template readArray<QualType>(buffer);
    } else if (isComputedNoexcept(esi.Type)) {
      esi.NoexceptExpr = asImpl().readExprRef();
    } else if (esi.Type == EST_Uninstantiated) {
      esi.SourceDecl = asImpl().readFunctionDeclRef();
      esi.SourceTemplate = asImpl().readFunctionDeclRef();
    } else if (esi.Type == EST_Unevaluated) {
      esi.SourceDecl = asImpl().readFunctionDeclRef();
    }
    return esi;
  }

  FunctionProtoType::ExtParameterInfo readExtParameterInfo() {
    static_assert(sizeof(FunctionProtoType::ExtParameterInfo().getOpaqueValue())
                    <= sizeof(uint32_t),
                  "opaque value doesn't fit into uint32_t");
    uint32_t value = asImpl().readUInt32();
    return FunctionProtoType::ExtParameterInfo::getFromOpaqueValue(value);
  }

  DeclarationName readDeclarationName() {
    auto &ctx = getASTContext();
    auto kind = asImpl().readDeclarationNameKind();
    switch (kind) {
    case DeclarationName::Identifier:
      return DeclarationName(asImpl().readIdentifier());

    case DeclarationName::ObjCZeroArgSelector:
    case DeclarationName::ObjCOneArgSelector:
    case DeclarationName::ObjCMultiArgSelector:
      return DeclarationName(asImpl().readSelector());

    case DeclarationName::CXXConstructorName:
      return ctx.DeclarationNames.getCXXConstructorName(
               ctx.getCanonicalType(asImpl().readQualType()));

    case DeclarationName::CXXDestructorName:
      return ctx.DeclarationNames.getCXXDestructorName(
               ctx.getCanonicalType(asImpl().readQualType()));

    case DeclarationName::CXXConversionFunctionName:
      return ctx.DeclarationNames.getCXXConversionFunctionName(
               ctx.getCanonicalType(asImpl().readQualType()));

    case DeclarationName::CXXDeductionGuideName:
      return ctx.DeclarationNames.getCXXDeductionGuideName(
               asImpl().readTemplateDeclRef());

    case DeclarationName::CXXOperatorName:
      return ctx.DeclarationNames.getCXXOperatorName(
               asImpl().readOverloadedOperatorKind());

    case DeclarationName::CXXLiteralOperatorName:
      return ctx.DeclarationNames.getCXXLiteralOperatorName(
               asImpl().readIdentifier());

    case DeclarationName::CXXUsingDirective:
      return DeclarationName::getUsingDirectiveName();
    }
    llvm_unreachable("bad name kind");
  }

  TemplateName readTemplateName() {
    auto &ctx = getASTContext();
    auto kind = asImpl().readTemplateNameKind();
    switch (kind) {
    case TemplateName::Template:
      return TemplateName(asImpl().readTemplateDeclRef());

    case TemplateName::OverloadedTemplate: {
      SmallVector<NamedDecl *, 8> buffer;
      auto overloadsArray = asImpl().template readArray<NamedDecl*>(buffer);

      // Copy into an UnresolvedSet to satisfy the interface.
      UnresolvedSet<8> overloads;
      for (auto overload : overloadsArray) {
        overloads.addDecl(overload);
      }

      return ctx.getOverloadedTemplateName(overloads.begin(), overloads.end());
    }

    case TemplateName::AssumedTemplate: {
      auto name = asImpl().readDeclarationName();
      return ctx.getAssumedTemplateName(name);
    }

    case TemplateName::QualifiedTemplate: {
      auto qual = asImpl().readNestedNameSpecifier();
      auto hasTemplateKeyword = asImpl().readBool();
      auto templateDecl = asImpl().readTemplateDeclRef();
      return ctx.getQualifiedTemplateName(qual, hasTemplateKeyword,
                                          templateDecl);
    }

    case TemplateName::DependentTemplate: {
      auto qual = asImpl().readNestedNameSpecifier();
      auto isIdentifier = asImpl().readBool();
      if (isIdentifier) {
        return ctx.getDependentTemplateName(qual, asImpl().readIdentifier());
      } else {
        return ctx.getDependentTemplateName(qual,
                 asImpl().readOverloadedOperatorKind());
      }
    }

    case TemplateName::SubstTemplateTemplateParm: {
      auto param = asImpl().readTemplateTemplateParmDeclRef();
      auto replacement = asImpl().readTemplateName();
      return ctx.getSubstTemplateTemplateParm(param, replacement);
    }

    case TemplateName::SubstTemplateTemplateParmPack: {
      auto param = asImpl().readTemplateTemplateParmDeclRef();
      auto replacement = asImpl().readTemplateName();
      return ctx.getSubstTemplateTemplateParmPack(param, replacement);
    }
    }
    llvm_unreachable("bad template name kind");
  }

  TemplateArgument readTemplateArgument(bool canonicalize = false) {
    if (canonicalize) {
      return getASTContext().getCanonicalTemplateArgument(
               readTemplateArgument(false));
    }

    auto kind = asImpl().readTemplateArgumentKind();
    switch (kind) {
    case TemplateArgument::Null:
      return TemplateArgument();
    case TemplateArgument::Type:
      return TemplateArgument(asImpl().readQualType());
    case TemplateArgument::Declaration: {
      auto decl = asImpl().readValueDeclRef();
      auto type = asImpl().readQualType();
      return TemplateArgument(decl, type);
    }
    case TemplateArgument::NullPtr:
      return TemplateArgument(asImpl().readQualType(), /*nullptr*/ true);
    case TemplateArgument::Integral: {
      auto value = asImpl().readAPSInt();
      auto type = asImpl().readQualType();
      return TemplateArgument(getASTContext(), value, type);
    }
    case TemplateArgument::Template:
      return TemplateArgument(asImpl().readTemplateName());
    case TemplateArgument::TemplateExpansion: {
      auto name = asImpl().readTemplateName();
      auto numExpansions = asImpl().template readOptional<uint32_t>();
      return TemplateArgument(name, numExpansions);
    }
    case TemplateArgument::Expression:
      return TemplateArgument(asImpl().readExprRef());
    case TemplateArgument::Pack: {
      llvm::SmallVector<TemplateArgument, 8> packBuffer;
      auto pack = asImpl().template readArray<TemplateArgument>(packBuffer);

      // Copy the pack into the ASTContext.
      TemplateArgument *contextPack =
        new (getASTContext()) TemplateArgument[pack.size()];
      for (size_t i = 0, e = pack.size(); i != e; ++i)
        contextPack[i] = pack[i];
      return TemplateArgument(llvm::makeArrayRef(contextPack, pack.size()));
    }
    }
    llvm_unreachable("bad template argument kind");
  }

  NestedNameSpecifier *readNestedNameSpecifier() {
    auto &ctx = getASTContext();

    // We build this up iteratively.
    NestedNameSpecifier *cur = nullptr;

    uint32_t depth = asImpl().readUInt32();
    for (uint32_t i = 0; i != depth; ++i) {
      auto kind = asImpl().readNestedNameSpecifierKind();
      switch (kind) {
      case NestedNameSpecifier::Identifier:
        cur = NestedNameSpecifier::Create(ctx, cur,
                                          asImpl().readIdentifier());
        continue;

      case NestedNameSpecifier::Namespace:
        cur = NestedNameSpecifier::Create(ctx, cur,
                                          asImpl().readNamespaceDeclRef());
        continue;

      case NestedNameSpecifier::NamespaceAlias:
        cur = NestedNameSpecifier::Create(ctx, cur,
                                     asImpl().readNamespaceAliasDeclRef());
        continue;

      case NestedNameSpecifier::TypeSpec:
      case NestedNameSpecifier::TypeSpecWithTemplate:
        cur = NestedNameSpecifier::Create(ctx, cur,
                          kind == NestedNameSpecifier::TypeSpecWithTemplate,
                          asImpl().readQualType().getTypePtr());
        continue;

      case NestedNameSpecifier::Global:
        cur = NestedNameSpecifier::GlobalSpecifier(ctx);
        continue;

      case NestedNameSpecifier::Super:
        cur = NestedNameSpecifier::SuperSpecifier(ctx,
                                            asImpl().readCXXRecordDeclRef());
        continue;
      }
      llvm_unreachable("bad nested name specifier kind");
    }

    return cur;
  }
};

} // end namespace serialization
} // end namespace clang

#endif
