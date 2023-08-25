//===- ExtractAPI/DeclarationFragments.cpp ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements Declaration Fragments related classes.
///
//===----------------------------------------------------------------------===//

#include "clang/ExtractAPI/DeclarationFragments.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/QualTypeNames.h"
#include "clang/Basic/OperatorKinds.h"
#include "clang/ExtractAPI/TypedefUnderlyingTypeResolver.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/StringSwitch.h"
#include <typeinfo>

using namespace clang::extractapi;
using namespace llvm;

DeclarationFragments &DeclarationFragments::appendSpace() {
  if (!Fragments.empty()) {
    Fragment &Last = Fragments.back();
    if (Last.Kind == FragmentKind::Text) {
      // Merge the extra space into the last fragment if the last fragment is
      // also text.
      if (Last.Spelling.back() != ' ') { // avoid extra trailing spaces.
        Last.Spelling.push_back(' ');
      }
    } else {
      append(" ", FragmentKind::Text);
    }
  }

  return *this;
}

StringRef DeclarationFragments::getFragmentKindString(
    DeclarationFragments::FragmentKind Kind) {
  switch (Kind) {
  case DeclarationFragments::FragmentKind::None:
    return "none";
  case DeclarationFragments::FragmentKind::Keyword:
    return "keyword";
  case DeclarationFragments::FragmentKind::Attribute:
    return "attribute";
  case DeclarationFragments::FragmentKind::NumberLiteral:
    return "number";
  case DeclarationFragments::FragmentKind::StringLiteral:
    return "string";
  case DeclarationFragments::FragmentKind::Identifier:
    return "identifier";
  case DeclarationFragments::FragmentKind::TypeIdentifier:
    return "typeIdentifier";
  case DeclarationFragments::FragmentKind::GenericParameter:
    return "genericParameter";
  case DeclarationFragments::FragmentKind::ExternalParam:
    return "externalParam";
  case DeclarationFragments::FragmentKind::InternalParam:
    return "internalParam";
  case DeclarationFragments::FragmentKind::Text:
    return "text";
  }

  llvm_unreachable("Unhandled FragmentKind");
}

DeclarationFragments::FragmentKind
DeclarationFragments::parseFragmentKindFromString(StringRef S) {
  return llvm::StringSwitch<FragmentKind>(S)
      .Case("keyword", DeclarationFragments::FragmentKind::Keyword)
      .Case("attribute", DeclarationFragments::FragmentKind::Attribute)
      .Case("number", DeclarationFragments::FragmentKind::NumberLiteral)
      .Case("string", DeclarationFragments::FragmentKind::StringLiteral)
      .Case("identifier", DeclarationFragments::FragmentKind::Identifier)
      .Case("typeIdentifier",
            DeclarationFragments::FragmentKind::TypeIdentifier)
      .Case("genericParameter",
            DeclarationFragments::FragmentKind::GenericParameter)
      .Case("internalParam", DeclarationFragments::FragmentKind::InternalParam)
      .Case("externalParam", DeclarationFragments::FragmentKind::ExternalParam)
      .Case("text", DeclarationFragments::FragmentKind::Text)
      .Default(DeclarationFragments::FragmentKind::None);
}

DeclarationFragments DeclarationFragments::getExceptionSpecificationString(
    ExceptionSpecificationType ExceptionSpec) {
  DeclarationFragments Fragments;
  switch (ExceptionSpec) {
  case ExceptionSpecificationType::EST_None:
    return Fragments;
  case ExceptionSpecificationType::EST_DynamicNone:
    return Fragments.append(" ", DeclarationFragments::FragmentKind::Text)
        .append("throw", DeclarationFragments::FragmentKind::Keyword)
        .append("(", DeclarationFragments::FragmentKind::Text)
        .append(")", DeclarationFragments::FragmentKind::Text);
  case ExceptionSpecificationType::EST_Dynamic:
    // FIXME: throw(int), get types of inner expression
    return Fragments;
  case ExceptionSpecificationType::EST_BasicNoexcept:
    return Fragments.append(" ", DeclarationFragments::FragmentKind::Text)
        .append("noexcept", DeclarationFragments::FragmentKind::Keyword);
  case ExceptionSpecificationType::EST_DependentNoexcept:
    // FIXME: throw(conditional-expression), get expression
    break;
  case ExceptionSpecificationType::EST_NoexceptFalse:
    return Fragments.append(" ", DeclarationFragments::FragmentKind::Text)
        .append("noexcept", DeclarationFragments::FragmentKind::Keyword)
        .append("(", DeclarationFragments::FragmentKind::Text)
        .append("false", DeclarationFragments::FragmentKind::Keyword)
        .append(")", DeclarationFragments::FragmentKind::Text);
  case ExceptionSpecificationType::EST_NoexceptTrue:
    return Fragments.append(" ", DeclarationFragments::FragmentKind::Text)
        .append("noexcept", DeclarationFragments::FragmentKind::Keyword)
        .append("(", DeclarationFragments::FragmentKind::Text)
        .append("true", DeclarationFragments::FragmentKind::Keyword)
        .append(")", DeclarationFragments::FragmentKind::Text);
  default:
    return Fragments;
  }

  llvm_unreachable("Unhandled exception specification");
}

DeclarationFragments
DeclarationFragments::getStructureTypeFragment(const RecordDecl *Record) {
  DeclarationFragments Fragments;
  if (Record->isStruct())
    Fragments.append("struct", DeclarationFragments::FragmentKind::Keyword);
  else if (Record->isUnion())
    Fragments.append("union", DeclarationFragments::FragmentKind::Keyword);
  else
    Fragments.append("class", DeclarationFragments::FragmentKind::Keyword);

  return Fragments;
}

// NNS stores C++ nested name specifiers, which are prefixes to qualified names.
// Build declaration fragments for NNS recursively so that we have the USR for
// every part in a qualified name, and also leaves the actual underlying type
// cleaner for its own fragment.
DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForNNS(const NestedNameSpecifier *NNS,
                                                ASTContext &Context,
                                                DeclarationFragments &After) {
  DeclarationFragments Fragments;
  if (NNS->getPrefix())
    Fragments.append(getFragmentsForNNS(NNS->getPrefix(), Context, After));

  switch (NNS->getKind()) {
  case NestedNameSpecifier::Identifier:
    Fragments.append(NNS->getAsIdentifier()->getName(),
                     DeclarationFragments::FragmentKind::Identifier);
    break;

  case NestedNameSpecifier::Namespace: {
    const NamespaceDecl *NS = NNS->getAsNamespace();
    if (NS->isAnonymousNamespace())
      return Fragments;
    SmallString<128> USR;
    index::generateUSRForDecl(NS, USR);
    Fragments.append(NS->getName(),
                     DeclarationFragments::FragmentKind::Identifier, USR, NS);
    break;
  }

  case NestedNameSpecifier::NamespaceAlias: {
    const NamespaceAliasDecl *Alias = NNS->getAsNamespaceAlias();
    SmallString<128> USR;
    index::generateUSRForDecl(Alias, USR);
    Fragments.append(Alias->getName(),
                     DeclarationFragments::FragmentKind::Identifier, USR,
                     Alias);
    break;
  }

  case NestedNameSpecifier::Global:
    // The global specifier `::` at the beginning. No stored value.
    break;

  case NestedNameSpecifier::Super:
    // Microsoft's `__super` specifier.
    Fragments.append("__super", DeclarationFragments::FragmentKind::Keyword);
    break;

  case NestedNameSpecifier::TypeSpecWithTemplate:
    // A type prefixed by the `template` keyword.
    Fragments.append("template", DeclarationFragments::FragmentKind::Keyword);
    Fragments.appendSpace();
    // Fallthrough after adding the keyword to handle the actual type.
    [[fallthrough]];

  case NestedNameSpecifier::TypeSpec: {
    const Type *T = NNS->getAsType();
    // FIXME: Handle C++ template specialization type
    Fragments.append(getFragmentsForType(T, Context, After));
    break;
  }
  }

  // Add the separator text `::` for this segment.
  return Fragments.append("::", DeclarationFragments::FragmentKind::Text);
}

// Recursively build the declaration fragments for an underlying `Type` with
// qualifiers removed.
DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForType(
    const Type *T, ASTContext &Context, DeclarationFragments &After) {
  assert(T && "invalid type");

  DeclarationFragments Fragments;

  // Declaration fragments of a pointer type is the declaration fragments of
  // the pointee type followed by a `*`,
  if (T->isPointerType())
    return Fragments
        .append(getFragmentsForType(T->getPointeeType(), Context, After))
        .append(" *", DeclarationFragments::FragmentKind::Text);

  // For Objective-C `id` and `Class` pointers
  // we do not spell out the `*`.
  if (T->isObjCObjectPointerType() &&
      !T->getAs<ObjCObjectPointerType>()->isObjCIdOrClassType()) {

    Fragments.append(getFragmentsForType(T->getPointeeType(), Context, After));

    // id<protocol> is an qualified id type
    // id<protocol>* is not an qualified id type
    if (!T->getAs<ObjCObjectPointerType>()->isObjCQualifiedIdType()) {
      Fragments.append(" *", DeclarationFragments::FragmentKind::Text);
    }

    return Fragments;
  }

  // Declaration fragments of a lvalue reference type is the declaration
  // fragments of the underlying type followed by a `&`.
  if (const LValueReferenceType *LRT = dyn_cast<LValueReferenceType>(T))
    return Fragments
        .append(
            getFragmentsForType(LRT->getPointeeTypeAsWritten(), Context, After))
        .append(" &", DeclarationFragments::FragmentKind::Text);

  // Declaration fragments of a rvalue reference type is the declaration
  // fragments of the underlying type followed by a `&&`.
  if (const RValueReferenceType *RRT = dyn_cast<RValueReferenceType>(T))
    return Fragments
        .append(
            getFragmentsForType(RRT->getPointeeTypeAsWritten(), Context, After))
        .append(" &&", DeclarationFragments::FragmentKind::Text);

  // Declaration fragments of an array-typed variable have two parts:
  // 1. the element type of the array that appears before the variable name;
  // 2. array brackets `[(0-9)?]` that appear after the variable name.
  if (const ArrayType *AT = T->getAsArrayTypeUnsafe()) {
    // Build the "after" part first because the inner element type might also
    // be an array-type. For example `int matrix[3][4]` which has a type of
    // "(array 3 of (array 4 of ints))."
    // Push the array size part first to make sure they are in the right order.
    After.append("[", DeclarationFragments::FragmentKind::Text);

    switch (AT->getSizeModifier()) {
    case ArrayType::Normal:
      break;
    case ArrayType::Static:
      Fragments.append("static", DeclarationFragments::FragmentKind::Keyword);
      break;
    case ArrayType::Star:
      Fragments.append("*", DeclarationFragments::FragmentKind::Text);
      break;
    }

    if (const ConstantArrayType *CAT = dyn_cast<ConstantArrayType>(AT)) {
      // FIXME: right now this would evaluate any expressions/macros written in
      // the original source to concrete values. For example
      // `int nums[MAX]` -> `int nums[100]`
      // `char *str[5 + 1]` -> `char *str[6]`
      SmallString<128> Size;
      CAT->getSize().toStringUnsigned(Size);
      After.append(Size, DeclarationFragments::FragmentKind::NumberLiteral);
    }

    After.append("]", DeclarationFragments::FragmentKind::Text);

    return Fragments.append(
        getFragmentsForType(AT->getElementType(), Context, After));
  }

  // An ElaboratedType is a sugar for types that are referred to using an
  // elaborated keyword, e.g., `struct S`, `enum E`, or (in C++) via a
  // qualified name, e.g., `N::M::type`, or both.
  if (const ElaboratedType *ET = dyn_cast<ElaboratedType>(T)) {
    ElaboratedTypeKeyword Keyword = ET->getKeyword();
    if (Keyword != ETK_None) {
      Fragments
          .append(ElaboratedType::getKeywordName(Keyword),
                  DeclarationFragments::FragmentKind::Keyword)
          .appendSpace();
    }

    if (const NestedNameSpecifier *NNS = ET->getQualifier())
      Fragments.append(getFragmentsForNNS(NNS, Context, After));

    // After handling the elaborated keyword or qualified name, build
    // declaration fragments for the desugared underlying type.
    return Fragments.append(getFragmentsForType(ET->desugar(), Context, After));
  }

  // If the type is a typedefed type, get the underlying TypedefNameDecl for a
  // direct reference to the typedef instead of the wrapped type.

  // 'id' type is a typedef for an ObjCObjectPointerType
  //  we treat it as a typedef
  if (const TypedefType *TypedefTy = dyn_cast<TypedefType>(T)) {
    const TypedefNameDecl *Decl = TypedefTy->getDecl();
    TypedefUnderlyingTypeResolver TypedefResolver(Context);
    std::string USR = TypedefResolver.getUSRForType(QualType(T, 0));

    if (T->isObjCIdType()) {
      return Fragments.append(Decl->getName(),
                              DeclarationFragments::FragmentKind::Keyword);
    }

    return Fragments.append(
        Decl->getName(), DeclarationFragments::FragmentKind::TypeIdentifier,
        USR, TypedefResolver.getUnderlyingTypeDecl(QualType(T, 0)));
  }

  // Everything we care about has been handled now, reduce to the canonical
  // unqualified base type.
  QualType Base = T->getCanonicalTypeUnqualified();

  // If the base type is a TagType (struct/interface/union/class/enum), let's
  // get the underlying Decl for better names and USRs.
  if (const TagType *TagTy = dyn_cast<TagType>(Base)) {
    const TagDecl *Decl = TagTy->getDecl();
    // Anonymous decl, skip this fragment.
    if (Decl->getName().empty())
      return Fragments;
    SmallString<128> TagUSR;
    clang::index::generateUSRForDecl(Decl, TagUSR);
    return Fragments.append(Decl->getName(),
                            DeclarationFragments::FragmentKind::TypeIdentifier,
                            TagUSR, Decl);
  }

  // If the base type is an ObjCInterfaceType, use the underlying
  // ObjCInterfaceDecl for the true USR.
  if (const auto *ObjCIT = dyn_cast<ObjCInterfaceType>(Base)) {
    const auto *Decl = ObjCIT->getDecl();
    SmallString<128> USR;
    index::generateUSRForDecl(Decl, USR);
    return Fragments.append(Decl->getName(),
                            DeclarationFragments::FragmentKind::TypeIdentifier,
                            USR, Decl);
  }

  // Default fragment builder for other kinds of types (BuiltinType etc.)
  SmallString<128> USR;
  clang::index::generateUSRForType(Base, Context, USR);
  Fragments.append(Base.getAsString(),
                   DeclarationFragments::FragmentKind::TypeIdentifier, USR);

  return Fragments;
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForQualifiers(const Qualifiers Quals) {
  DeclarationFragments Fragments;
  if (Quals.hasConst())
    Fragments.append("const", DeclarationFragments::FragmentKind::Keyword);
  if (Quals.hasVolatile())
    Fragments.append("volatile", DeclarationFragments::FragmentKind::Keyword);
  if (Quals.hasRestrict())
    Fragments.append("restrict", DeclarationFragments::FragmentKind::Keyword);

  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForType(
    const QualType QT, ASTContext &Context, DeclarationFragments &After) {
  assert(!QT.isNull() && "invalid type");

  if (const ParenType *PT = dyn_cast<ParenType>(QT)) {
    After.append(")", DeclarationFragments::FragmentKind::Text);
    return getFragmentsForType(PT->getInnerType(), Context, After)
        .append("(", DeclarationFragments::FragmentKind::Text);
  }

  const SplitQualType SQT = QT.split();
  DeclarationFragments QualsFragments = getFragmentsForQualifiers(SQT.Quals),
                       TypeFragments =
                           getFragmentsForType(SQT.Ty, Context, After);
  if (QT.getAsString() == "_Bool")
    TypeFragments.replace("bool", 0);

  if (QualsFragments.getFragments().empty())
    return TypeFragments;

  // Use east qualifier for pointer types
  // For example:
  // ```
  // int *   const
  // ^----   ^----
  //  type    qualifier
  // ^-----------------
  //  const pointer to int
  // ```
  // should not be reconstructed as
  // ```
  // const       int       *
  // ^----       ^--
  //  qualifier   type
  // ^----------------     ^
  //  pointer to const int
  // ```
  if (SQT.Ty->isAnyPointerType())
    return TypeFragments.appendSpace().append(std::move(QualsFragments));

  return QualsFragments.appendSpace().append(std::move(TypeFragments));
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForNamespace(
    const NamespaceDecl *Decl) {
  DeclarationFragments Fragments;
  Fragments.append("namespace", DeclarationFragments::FragmentKind::Keyword);
  if (!Decl->isAnonymousNamespace())
    Fragments.appendSpace().append(
        Decl->getName(), DeclarationFragments::FragmentKind::Identifier);
  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForVar(const VarDecl *Var) {
  DeclarationFragments Fragments;
  if (Var->isConstexpr())
    Fragments.append("constexpr", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();

  StorageClass SC = Var->getStorageClass();
  if (SC != SC_None)
    Fragments
        .append(VarDecl::getStorageClassSpecifierString(SC),
                DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  QualType T =
      Var->getTypeSourceInfo()
          ? Var->getTypeSourceInfo()->getType()
          : Var->getASTContext().getUnqualifiedObjCPointerType(Var->getType());

  // Capture potential fragments that needs to be placed after the variable name
  // ```
  // int nums[5];
  // char (*ptr_to_array)[6];
  // ```
  DeclarationFragments After;
  return Fragments.append(getFragmentsForType(T, Var->getASTContext(), After))
      .appendSpace()
      .append(Var->getName(), DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After))
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForVarTemplate(const VarDecl *Var) {
  DeclarationFragments Fragments;
  if (Var->isConstexpr())
    Fragments.append("constexpr", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  QualType T =
      Var->getTypeSourceInfo()
          ? Var->getTypeSourceInfo()->getType()
          : Var->getASTContext().getUnqualifiedObjCPointerType(Var->getType());

  // Might be a member, so might be static.
  if (Var->isStaticDataMember())
    Fragments.append("static", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();

  DeclarationFragments After;
  DeclarationFragments ArgumentFragment =
      getFragmentsForType(T, Var->getASTContext(), After);
  if (ArgumentFragment.begin()->Spelling.substr(0, 14).compare(
          "type-parameter") == 0) {
    std::string ProperArgName = getNameForTemplateArgument(
        Var->getDescribedVarTemplate()->getTemplateParameters()->asArray(),
        ArgumentFragment.begin()->Spelling);
    ArgumentFragment.begin()->Spelling.swap(ProperArgName);
  }
  Fragments.append(std::move(ArgumentFragment))
      .appendSpace()
      .append(Var->getName(), DeclarationFragments::FragmentKind::Identifier)
      .append(";", DeclarationFragments::FragmentKind::Text);
  return Fragments;
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForParam(const ParmVarDecl *Param) {
  DeclarationFragments Fragments, After;

  QualType T = Param->getTypeSourceInfo()
                   ? Param->getTypeSourceInfo()->getType()
                   : Param->getASTContext().getUnqualifiedObjCPointerType(
                         Param->getType());

  DeclarationFragments TypeFragments =
      getFragmentsForType(T, Param->getASTContext(), After);
  if (TypeFragments.begin()->Spelling.substr(0, 14).compare("type-parameter") ==
      0) {
    std::string ProperArgName = getNameForTemplateArgument(
        dyn_cast<FunctionDecl>(Param->getDeclContext())
            ->getDescribedFunctionTemplate()
            ->getTemplateParameters()
            ->asArray(),
        TypeFragments.begin()->Spelling);
    TypeFragments.begin()->Spelling.swap(ProperArgName);
  }

  if (Param->isObjCMethodParameter())
    Fragments.append("(", DeclarationFragments::FragmentKind::Text)
        .append(std::move(TypeFragments))
        .append(") ", DeclarationFragments::FragmentKind::Text);
  else
    Fragments.append(std::move(TypeFragments)).appendSpace();

  return Fragments
      .append(Param->getName(),
              DeclarationFragments::FragmentKind::InternalParam)
      .append(std::move(After));
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForFunction(const FunctionDecl *Func) {
  DeclarationFragments Fragments;
  // FIXME: Handle template specialization
  switch (Func->getStorageClass()) {
  case SC_None:
  case SC_PrivateExtern:
    break;
  case SC_Extern:
    Fragments.append("extern", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
    break;
  case SC_Static:
    Fragments.append("static", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
    break;
  case SC_Auto:
  case SC_Register:
    llvm_unreachable("invalid for functions");
  }
  if (Func->isConsteval()) // if consteval, it is also constexpr
    Fragments.append("consteval", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  else if (Func->isConstexpr())
    Fragments.append("constexpr", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();

  // FIXME: Is `after` actually needed here?
  DeclarationFragments After;
  auto ReturnValueFragment =
      getFragmentsForType(Func->getReturnType(), Func->getASTContext(), After);
  if (ReturnValueFragment.begin()->Spelling.substr(0, 14).compare(
          "type-parameter") == 0) {
    std::string ProperArgName =
        getNameForTemplateArgument(Func->getDescribedFunctionTemplate()
                                       ->getTemplateParameters()
                                       ->asArray(),
                                   ReturnValueFragment.begin()->Spelling);
    ReturnValueFragment.begin()->Spelling.swap(ProperArgName);
  }

  Fragments.append(std::move(ReturnValueFragment))
      .appendSpace()
      .append(Func->getName(), DeclarationFragments::FragmentKind::Identifier);

  if (Func->getTemplateSpecializationInfo()) {
    Fragments.append("<", DeclarationFragments::FragmentKind::Text);

    for (unsigned i = 0, end = Func->getNumParams(); i != end; ++i) {
      if (i)
        Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
      Fragments.append(
          getFragmentsForType(Func->getParamDecl(i)->getType(),
                              Func->getParamDecl(i)->getASTContext(), After));
    }
    Fragments.append(">", DeclarationFragments::FragmentKind::Text);
  }
  Fragments.append(std::move(After));

  Fragments.append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = Func->getNumParams(); i != end; ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(Func->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  Fragments.append(DeclarationFragments::getExceptionSpecificationString(
      Func->getExceptionSpecType()));

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForEnumConstant(
    const EnumConstantDecl *EnumConstDecl) {
  DeclarationFragments Fragments;
  return Fragments.append(EnumConstDecl->getName(),
                          DeclarationFragments::FragmentKind::Identifier);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForEnum(const EnumDecl *EnumDecl) {
  if (const auto *TypedefNameDecl = EnumDecl->getTypedefNameForAnonDecl())
    return getFragmentsForTypedef(TypedefNameDecl);

  DeclarationFragments Fragments, After;
  Fragments.append("enum", DeclarationFragments::FragmentKind::Keyword);

  if (!EnumDecl->getName().empty())
    Fragments.appendSpace().append(
        EnumDecl->getName(), DeclarationFragments::FragmentKind::Identifier);

  QualType IntegerType = EnumDecl->getIntegerType();
  if (!IntegerType.isNull())
    Fragments.append(": ", DeclarationFragments::FragmentKind::Text)
        .append(
            getFragmentsForType(IntegerType, EnumDecl->getASTContext(), After))
        .append(std::move(After));

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForField(const FieldDecl *Field) {
  DeclarationFragments After;
  DeclarationFragments Fragments;
  if (Field->isMutable())
    Fragments.append("mutable", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  return Fragments
      .append(
          getFragmentsForType(Field->getType(), Field->getASTContext(), After))
      .appendSpace()
      .append(Field->getName(), DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After))
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForStruct(const RecordDecl *Record) {
  if (const auto *TypedefNameDecl = Record->getTypedefNameForAnonDecl())
    return getFragmentsForTypedef(TypedefNameDecl);

  DeclarationFragments Fragments;
  Fragments.append("struct", DeclarationFragments::FragmentKind::Keyword);

  if (!Record->getName().empty())
    Fragments.appendSpace().append(
        Record->getName(), DeclarationFragments::FragmentKind::Identifier);

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForCXXClass(
    const CXXRecordDecl *Record) {
  if (const auto *TypedefNameDecl = Record->getTypedefNameForAnonDecl())
    return getFragmentsForTypedef(TypedefNameDecl);

  DeclarationFragments Fragments;
  Fragments.append(DeclarationFragments::getStructureTypeFragment(Record));

  if (!Record->getName().empty())
    Fragments.appendSpace().append(
        Record->getName(), DeclarationFragments::FragmentKind::Identifier);

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForSpecialCXXMethod(
    const CXXMethodDecl *Method) {
  DeclarationFragments Fragments;
  std::string Name;
  if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(Method)) {
    Name = Method->getNameAsString();
    if (Constructor->isExplicit())
      Fragments.append("explicit", DeclarationFragments::FragmentKind::Keyword)
          .appendSpace();
  } else if (isa<CXXDestructorDecl>(Method))
    Name = Method->getNameAsString();

  DeclarationFragments After;
  Fragments.append(Name, DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After));
  Fragments.append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = Method->getNumParams(); i != end; ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(Method->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  Fragments.append(DeclarationFragments::getExceptionSpecificationString(
      Method->getExceptionSpecType()));

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForCXXMethod(
    const CXXMethodDecl *Method) {
  DeclarationFragments Fragments;
  StringRef Name = Method->getName();
  if (Method->isStatic())
    Fragments.append("static", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  if (Method->isConstexpr())
    Fragments.append("constexpr", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();
  if (Method->isVolatile())
    Fragments.append("volatile", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();

  // Build return type
  DeclarationFragments After;
  Fragments
      .append(getFragmentsForType(Method->getReturnType(),
                                  Method->getASTContext(), After))
      .appendSpace()
      .append(Name, DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After));
  Fragments.append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = Method->getNumParams(); i != end; ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(Method->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  if (Method->isConst())
    Fragments.appendSpace().append("const",
                                   DeclarationFragments::FragmentKind::Keyword);

  Fragments.append(DeclarationFragments::getExceptionSpecificationString(
      Method->getExceptionSpecType()));

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForConversionFunction(
    const CXXConversionDecl *ConversionFunction) {
  DeclarationFragments Fragments;

  if (ConversionFunction->isExplicit())
    Fragments.append("explicit", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace();

  Fragments.append("operator", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace();

  Fragments
      .append(ConversionFunction->getConversionType().getAsString(),
              DeclarationFragments::FragmentKind::TypeIdentifier)
      .append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = ConversionFunction->getNumParams(); i != end;
       ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(ConversionFunction->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  if (ConversionFunction->isConst())
    Fragments.appendSpace().append("const",
                                   DeclarationFragments::FragmentKind::Keyword);

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForOverloadedOperator(
    const CXXMethodDecl *Method) {
  DeclarationFragments Fragments;

  // Build return type
  DeclarationFragments After;
  Fragments
      .append(getFragmentsForType(Method->getReturnType(),
                                  Method->getASTContext(), After))
      .appendSpace()
      .append(Method->getNameAsString(),
              DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After));
  Fragments.append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = Method->getNumParams(); i != end; ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(Method->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  if (Method->isConst())
    Fragments.appendSpace().append("const",
                                   DeclarationFragments::FragmentKind::Keyword);

  Fragments.append(DeclarationFragments::getExceptionSpecificationString(
      Method->getExceptionSpecType()));

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

// Get fragments for template parameters, e.g. T in tempalte<typename T> ...
DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForTemplateParameters(
    ArrayRef<NamedDecl *> ParameterArray) {
  DeclarationFragments Fragments;
  for (unsigned i = 0, end = ParameterArray.size(); i != end; ++i) {
    if (i)
      Fragments.append(",", DeclarationFragments::FragmentKind::Text)
          .appendSpace();

    const auto *TemplateParam =
        dyn_cast<TemplateTypeParmDecl>(ParameterArray[i]);
    if (!TemplateParam)
      continue;
    if (TemplateParam->hasTypeConstraint())
      Fragments.append(TemplateParam->getTypeConstraint()
                           ->getNamedConcept()
                           ->getName()
                           .str(),
                       DeclarationFragments::FragmentKind::TypeIdentifier);
    else if (TemplateParam->wasDeclaredWithTypename())
      Fragments.append("typename", DeclarationFragments::FragmentKind::Keyword);
    else
      Fragments.append("class", DeclarationFragments::FragmentKind::Keyword);

    if (TemplateParam->isParameterPack())
      Fragments.append("...", DeclarationFragments::FragmentKind::Text);

    Fragments.appendSpace().append(
        TemplateParam->getName(),
        DeclarationFragments::FragmentKind::GenericParameter);
  }
  return Fragments;
}

// Find the name of a template argument from the template's parameters.
std::string DeclarationFragmentsBuilder::getNameForTemplateArgument(
    const ArrayRef<NamedDecl *> TemplateParameters, std::string TypeParameter) {
  // The arg is a generic parameter from a partial spec, e.g.
  // T in template<typename T> Foo<T, int>.
  //
  // Those names appear as "type-parameter-<index>-<depth>", so we must find its
  // name from the template's parameter list.
  for (unsigned i = 0; i < TemplateParameters.size(); ++i) {
    const auto *Parameter =
        dyn_cast<TemplateTypeParmDecl>(TemplateParameters[i]);
    if (TypeParameter.compare("type-parameter-" +
                              std::to_string(Parameter->getDepth()) + "-" +
                              std::to_string(Parameter->getIndex())) == 0)
      return std::string(TemplateParameters[i]->getName());
  }
  llvm_unreachable("Could not find the name of a template argument.");
}

// Get fragments for template arguments, e.g. int in template<typename T>
// Foo<int>;
//
// Note: TemplateParameters is only necessary if the Decl is a
// PartialSpecialization, where we need the parameters to deduce the name of the
// generic arguments.
DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForTemplateArguments(
    const ArrayRef<TemplateArgument> TemplateArguments, ASTContext &Context,
    const std::optional<ArrayRef<NamedDecl *>> TemplateParameters) {
  DeclarationFragments Fragments;
  for (unsigned i = 0, end = TemplateArguments.size(); i != end; ++i) {
    if (i)
      Fragments.append(",", DeclarationFragments::FragmentKind::Text)
          .appendSpace();

    std::string Type = TemplateArguments[i].getAsType().getAsString();
    DeclarationFragments After;
    DeclarationFragments ArgumentFragment =
        getFragmentsForType(TemplateArguments[i].getAsType(), Context, After);

    if (ArgumentFragment.begin()->Spelling.substr(0, 14).compare(
            "type-parameter") == 0) {
      std::string ProperArgName = getNameForTemplateArgument(
          TemplateParameters.value(), ArgumentFragment.begin()->Spelling);
      ArgumentFragment.begin()->Spelling.swap(ProperArgName);
    }
    Fragments.append(std::move(ArgumentFragment));

    if (TemplateArguments[i].isPackExpansion())
      Fragments.append("...", DeclarationFragments::FragmentKind::Text);
  }
  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForConcept(
    const ConceptDecl *Concept) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForTemplateParameters(
          Concept->getTemplateParameters()->asArray()))
      .append("> ", DeclarationFragments::FragmentKind::Text)
      .append("concept", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace()
      .append(Concept->getName().str(),
              DeclarationFragments::FragmentKind::Identifier)
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForRedeclarableTemplate(
    const RedeclarableTemplateDecl *RedeclarableTemplate) {
  DeclarationFragments Fragments;
  Fragments.append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForTemplateParameters(
          RedeclarableTemplate->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace();

  if (isa<TypeAliasTemplateDecl>(RedeclarableTemplate))
    Fragments.appendSpace()
        .append("using", DeclarationFragments::FragmentKind::Keyword)
        .appendSpace()
        .append(RedeclarableTemplate->getName(),
                DeclarationFragments::FragmentKind::Identifier);
  // the templated records will be resposbible for injecting their templates
  return Fragments.appendSpace();
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForClassTemplateSpecialization(
    const ClassTemplateSpecializationDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForCXXClass(
          cast<CXXRecordDecl>(Decl)))
      .pop_back() // there is an extra semicolon now
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(
          getFragmentsForTemplateArguments(Decl->getTemplateArgs().asArray(),
                                           Decl->getASTContext(), std::nullopt))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForClassTemplatePartialSpecialization(
    const ClassTemplatePartialSpecializationDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForTemplateParameters(
          Decl->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForCXXClass(
          cast<CXXRecordDecl>(Decl)))
      .pop_back() // there is an extra semicolon now
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForTemplateArguments(
          Decl->getTemplateArgs().asArray(), Decl->getASTContext(),
          Decl->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForVarTemplateSpecialization(
    const VarTemplateSpecializationDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForVarTemplate(Decl))
      .pop_back() // there is an extra semicolon now
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(
          getFragmentsForTemplateArguments(Decl->getTemplateArgs().asArray(),
                                           Decl->getASTContext(), std::nullopt))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForVarTemplatePartialSpecialization(
    const VarTemplatePartialSpecializationDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      // Partial specs may have new params.
      .append(getFragmentsForTemplateParameters(
          Decl->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForVarTemplate(Decl))
      .pop_back() // there is an extra semicolon now
      .append("<", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForTemplateArguments(
          Decl->getTemplateArgs().asArray(), Decl->getASTContext(),
          Decl->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForFunctionTemplate(
    const FunctionTemplateDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<", DeclarationFragments::FragmentKind::Text)
      // Partial specs may have new params.
      .append(getFragmentsForTemplateParameters(
          Decl->getTemplateParameters()->asArray()))
      .append(">", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForFunction(
          Decl->getAsFunction()));
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForFunctionTemplateSpecialization(
    const FunctionDecl *Decl) {
  DeclarationFragments Fragments;
  return Fragments
      .append("template", DeclarationFragments::FragmentKind::Keyword)
      .append("<>", DeclarationFragments::FragmentKind::Text)
      .appendSpace()
      .append(DeclarationFragmentsBuilder::getFragmentsForFunction(Decl));
}

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForMacro(StringRef Name,
                                                  const MacroDirective *MD) {
  DeclarationFragments Fragments;
  Fragments.append("#define", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace();
  Fragments.append(Name, DeclarationFragments::FragmentKind::Identifier);

  auto *MI = MD->getMacroInfo();

  if (MI->isFunctionLike()) {
    Fragments.append("(", DeclarationFragments::FragmentKind::Text);
    unsigned numParameters = MI->getNumParams();
    if (MI->isC99Varargs())
      --numParameters;
    for (unsigned i = 0; i < numParameters; ++i) {
      if (i)
        Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
      Fragments.append(MI->params()[i]->getName(),
                       DeclarationFragments::FragmentKind::InternalParam);
    }
    if (MI->isVariadic()) {
      if (numParameters && MI->isC99Varargs())
        Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
      Fragments.append("...", DeclarationFragments::FragmentKind::Text);
    }
    Fragments.append(")", DeclarationFragments::FragmentKind::Text);
  }
  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForObjCCategory(
    const ObjCCategoryDecl *Category) {
  DeclarationFragments Fragments;

  auto *Interface = Category->getClassInterface();
  SmallString<128> InterfaceUSR;
  index::generateUSRForDecl(Interface, InterfaceUSR);

  Fragments.append("@interface", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace()
      .append(Category->getClassInterface()->getName(),
              DeclarationFragments::FragmentKind::TypeIdentifier, InterfaceUSR,
              Interface)
      .append(" (", DeclarationFragments::FragmentKind::Text)
      .append(Category->getName(),
              DeclarationFragments::FragmentKind::Identifier)
      .append(")", DeclarationFragments::FragmentKind::Text);

  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForObjCInterface(
    const ObjCInterfaceDecl *Interface) {
  DeclarationFragments Fragments;
  // Build the base of the Objective-C interface declaration.
  Fragments.append("@interface", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace()
      .append(Interface->getName(),
              DeclarationFragments::FragmentKind::Identifier);

  // Build the inheritance part of the declaration.
  if (const ObjCInterfaceDecl *SuperClass = Interface->getSuperClass()) {
    SmallString<128> SuperUSR;
    index::generateUSRForDecl(SuperClass, SuperUSR);
    Fragments.append(" : ", DeclarationFragments::FragmentKind::Text)
        .append(SuperClass->getName(),
                DeclarationFragments::FragmentKind::TypeIdentifier, SuperUSR,
                SuperClass);
  }

  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForObjCMethod(
    const ObjCMethodDecl *Method) {
  DeclarationFragments Fragments, After;
  // Build the instance/class method indicator.
  if (Method->isClassMethod())
    Fragments.append("+ ", DeclarationFragments::FragmentKind::Text);
  else if (Method->isInstanceMethod())
    Fragments.append("- ", DeclarationFragments::FragmentKind::Text);

  // Build the return type.
  Fragments.append("(", DeclarationFragments::FragmentKind::Text)
      .append(getFragmentsForType(Method->getReturnType(),
                                  Method->getASTContext(), After))
      .append(std::move(After))
      .append(")", DeclarationFragments::FragmentKind::Text);

  // Build the selector part.
  Selector Selector = Method->getSelector();
  if (Selector.getNumArgs() == 0)
    // For Objective-C methods that don't take arguments, the first (and only)
    // slot of the selector is the method name.
    Fragments.appendSpace().append(
        Selector.getNameForSlot(0),
        DeclarationFragments::FragmentKind::Identifier);

  // For Objective-C methods that take arguments, build the selector slots.
  for (unsigned i = 0, end = Method->param_size(); i != end; ++i) {
    // Objective-C method selector parts are considered as identifiers instead
    // of "external parameters" as in Swift. This is because Objective-C method
    // symbols are referenced with the entire selector, instead of just the
    // method name in Swift.
    SmallString<32> ParamID(Selector.getNameForSlot(i));
    ParamID.append(":");
    Fragments.appendSpace().append(
        ParamID, DeclarationFragments::FragmentKind::Identifier);

    // Build the internal parameter.
    const ParmVarDecl *Param = Method->getParamDecl(i);
    Fragments.append(getFragmentsForParam(Param));
  }

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForObjCProperty(
    const ObjCPropertyDecl *Property) {
  DeclarationFragments Fragments, After;

  // Build the Objective-C property keyword.
  Fragments.append("@property", DeclarationFragments::FragmentKind::Keyword);

  const auto Attributes = Property->getPropertyAttributesAsWritten();
  // Build the attributes if there is any associated with the property.
  if (Attributes != ObjCPropertyAttribute::kind_noattr) {
    // No leading comma for the first attribute.
    bool First = true;
    Fragments.append(" (", DeclarationFragments::FragmentKind::Text);
    // Helper function to render the attribute.
    auto RenderAttribute =
        [&](ObjCPropertyAttribute::Kind Kind, StringRef Spelling,
            StringRef Arg = "",
            DeclarationFragments::FragmentKind ArgKind =
                DeclarationFragments::FragmentKind::Identifier) {
          // Check if the `Kind` attribute is set for this property.
          if ((Attributes & Kind) && !Spelling.empty()) {
            // Add a leading comma if this is not the first attribute rendered.
            if (!First)
              Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
            // Render the spelling of this attribute `Kind` as a keyword.
            Fragments.append(Spelling,
                             DeclarationFragments::FragmentKind::Keyword);
            // If this attribute takes in arguments (e.g. `getter=getterName`),
            // render the arguments.
            if (!Arg.empty())
              Fragments.append("=", DeclarationFragments::FragmentKind::Text)
                  .append(Arg, ArgKind);
            First = false;
          }
        };

    // Go through all possible Objective-C property attributes and render set
    // ones.
    RenderAttribute(ObjCPropertyAttribute::kind_class, "class");
    RenderAttribute(ObjCPropertyAttribute::kind_direct, "direct");
    RenderAttribute(ObjCPropertyAttribute::kind_nonatomic, "nonatomic");
    RenderAttribute(ObjCPropertyAttribute::kind_atomic, "atomic");
    RenderAttribute(ObjCPropertyAttribute::kind_assign, "assign");
    RenderAttribute(ObjCPropertyAttribute::kind_retain, "retain");
    RenderAttribute(ObjCPropertyAttribute::kind_strong, "strong");
    RenderAttribute(ObjCPropertyAttribute::kind_copy, "copy");
    RenderAttribute(ObjCPropertyAttribute::kind_weak, "weak");
    RenderAttribute(ObjCPropertyAttribute::kind_unsafe_unretained,
                    "unsafe_unretained");
    RenderAttribute(ObjCPropertyAttribute::kind_readwrite, "readwrite");
    RenderAttribute(ObjCPropertyAttribute::kind_readonly, "readonly");
    RenderAttribute(ObjCPropertyAttribute::kind_getter, "getter",
                    Property->getGetterName().getAsString());
    RenderAttribute(ObjCPropertyAttribute::kind_setter, "setter",
                    Property->getSetterName().getAsString());

    // Render nullability attributes.
    if (Attributes & ObjCPropertyAttribute::kind_nullability) {
      QualType Type = Property->getType();
      if (const auto Nullability =
              AttributedType::stripOuterNullability(Type)) {
        if (!First)
          Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
        if (*Nullability == NullabilityKind::Unspecified &&
            (Attributes & ObjCPropertyAttribute::kind_null_resettable))
          Fragments.append("null_resettable",
                           DeclarationFragments::FragmentKind::Keyword);
        else
          Fragments.append(
              getNullabilitySpelling(*Nullability, /*isContextSensitive=*/true),
              DeclarationFragments::FragmentKind::Keyword);
        First = false;
      }
    }

    Fragments.append(")", DeclarationFragments::FragmentKind::Text);
  }

  // Build the property type and name, and return the completed fragments.
  return Fragments.appendSpace()
      .append(getFragmentsForType(Property->getType(),
                                  Property->getASTContext(), After))
      .appendSpace()
      .append(Property->getName(),
              DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After));
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForObjCProtocol(
    const ObjCProtocolDecl *Protocol) {
  DeclarationFragments Fragments;
  // Build basic protocol declaration.
  Fragments.append("@protocol", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace()
      .append(Protocol->getName(),
              DeclarationFragments::FragmentKind::Identifier);

  // If this protocol conforms to other protocols, build the conformance list.
  if (!Protocol->protocols().empty()) {
    Fragments.append(" <", DeclarationFragments::FragmentKind::Text);
    for (ObjCProtocolDecl::protocol_iterator It = Protocol->protocol_begin();
         It != Protocol->protocol_end(); It++) {
      // Add a leading comma if this is not the first protocol rendered.
      if (It != Protocol->protocol_begin())
        Fragments.append(", ", DeclarationFragments::FragmentKind::Text);

      SmallString<128> USR;
      index::generateUSRForDecl(*It, USR);
      Fragments.append((*It)->getName(),
                       DeclarationFragments::FragmentKind::TypeIdentifier, USR,
                       *It);
    }
    Fragments.append(">", DeclarationFragments::FragmentKind::Text);
  }

  return Fragments;
}

DeclarationFragments DeclarationFragmentsBuilder::getFragmentsForTypedef(
    const TypedefNameDecl *Decl) {
  DeclarationFragments Fragments, After;
  Fragments.append("typedef", DeclarationFragments::FragmentKind::Keyword)
      .appendSpace()
      .append(getFragmentsForType(Decl->getUnderlyingType(),
                                  Decl->getASTContext(), After))
      .append(std::move(After))
      .appendSpace()
      .append(Decl->getName(), DeclarationFragments::FragmentKind::Identifier);

  return Fragments.append(";", DeclarationFragments::FragmentKind::Text);
}

// Instantiate template for FunctionDecl.
template FunctionSignature
DeclarationFragmentsBuilder::getFunctionSignature(const FunctionDecl *);

// Instantiate template for ObjCMethodDecl.
template FunctionSignature
DeclarationFragmentsBuilder::getFunctionSignature(const ObjCMethodDecl *);

// Subheading of a symbol defaults to its name.
DeclarationFragments
DeclarationFragmentsBuilder::getSubHeading(const NamedDecl *Decl) {
  DeclarationFragments Fragments;
  if (isa<CXXConstructorDecl>(Decl) || isa<CXXDestructorDecl>(Decl))
    Fragments.append(cast<CXXRecordDecl>(Decl->getDeclContext())->getName(),
                     DeclarationFragments::FragmentKind::Identifier);
  else if (isa<CXXConversionDecl>(Decl)) {
    Fragments.append(
        cast<CXXConversionDecl>(Decl)->getConversionType().getAsString(),
        DeclarationFragments::FragmentKind::Identifier);
  } else if (isa<CXXMethodDecl>(Decl) &&
             cast<CXXMethodDecl>(Decl)->isOverloadedOperator()) {
    Fragments.append(Decl->getNameAsString(),
                     DeclarationFragments::FragmentKind::Identifier);
  } else if (!Decl->getName().empty())
    Fragments.append(Decl->getName(),
                     DeclarationFragments::FragmentKind::Identifier);
  return Fragments;
}

// Subheading of an Objective-C method is a `+` or `-` sign indicating whether
// it's a class method or an instance method, followed by the selector name.
DeclarationFragments
DeclarationFragmentsBuilder::getSubHeading(const ObjCMethodDecl *Method) {
  DeclarationFragments Fragments;
  if (Method->isClassMethod())
    Fragments.append("+ ", DeclarationFragments::FragmentKind::Text);
  else if (Method->isInstanceMethod())
    Fragments.append("- ", DeclarationFragments::FragmentKind::Text);

  return Fragments.append(Method->getNameAsString(),
                          DeclarationFragments::FragmentKind::Identifier);
}

// Subheading of a symbol defaults to its name.
DeclarationFragments
DeclarationFragmentsBuilder::getSubHeadingForMacro(StringRef Name) {
  DeclarationFragments Fragments;
  Fragments.append(Name, DeclarationFragments::FragmentKind::Identifier);
  return Fragments;
}
