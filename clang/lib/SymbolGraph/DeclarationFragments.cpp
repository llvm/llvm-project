//===- SymbolGraph/DeclarationFragments.cpp ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines SymbolGraph Declaration Fragments related classes.
///
//===----------------------------------------------------------------------===//

#include "clang/SymbolGraph/DeclarationFragments.h"
#include "clang/Index/USRGeneration.h"
#include "llvm/ADT/StringSwitch.h"

namespace clang {
namespace symbolgraph {

DeclarationFragments &DeclarationFragments::appendSpace() {
  if (!Fragments.empty()) {
    Fragment Last = Fragments.back();
    if (Last.Kind == FragmentKind::Text) {
      if (Last.Spelling.back() != ' ') {
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
                     DeclarationFragments::FragmentKind::Identifier, USR);
    break;
  }

  case NestedNameSpecifier::NamespaceAlias: {
    const NamespaceAliasDecl *Alias = NNS->getAsNamespaceAlias();
    SmallString<128> USR;
    index::generateUSRForDecl(Alias, USR);
    Fragments.append(Alias->getName(),
                     DeclarationFragments::FragmentKind::Identifier, USR);
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
    LLVM_FALLTHROUGH;

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
  // the pointee type followed by a `*`, except for Objective-C `id` and `Class`
  // pointers, where we do not spell out the `*`.
  if (T->isPointerType() ||
      (T->isObjCObjectPointerType() &&
       !T->getAs<ObjCObjectPointerType>()->isObjCIdOrClassType())) {
    return Fragments
        .append(getFragmentsForType(T->getPointeeType(), Context, After))
        .append(" *", DeclarationFragments::FragmentKind::Text);
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

  // Everything we care about has been handled now, reduce to the canonical
  // unqualified base type.
  QualType Base = T->getCanonicalTypeUnqualified();

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

DeclarationFragments
DeclarationFragmentsBuilder::getFragmentsForVar(const VarDecl *Var) {
  DeclarationFragments Fragments;
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
      .append(std::move(After));
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

  if (Param->isObjCMethodParameter())
    Fragments.append("(", DeclarationFragments::FragmentKind::Text)
        .append(std::move(TypeFragments))
        .append(")", DeclarationFragments::FragmentKind::Text);
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
  // FIXME: Handle C++ function specifiers: constexpr, consteval, explicit, etc.

  // FIXME: Is `after` actually needed here?
  DeclarationFragments After;
  Fragments
      .append(getFragmentsForType(Func->getReturnType(), Func->getASTContext(),
                                  After))
      .appendSpace()
      .append(Func->getName(), DeclarationFragments::FragmentKind::Identifier)
      .append(std::move(After));

  Fragments.append("(", DeclarationFragments::FragmentKind::Text);
  for (unsigned i = 0, end = Func->getNumParams(); i != end; ++i) {
    if (i)
      Fragments.append(", ", DeclarationFragments::FragmentKind::Text);
    Fragments.append(getFragmentsForParam(Func->getParamDecl(i)));
  }
  Fragments.append(")", DeclarationFragments::FragmentKind::Text);

  // FIXME: Handle exception specifiers: throw, noexcept
  return Fragments;
}

FunctionSignature
DeclarationFragmentsBuilder::getFunctionSignature(const FunctionDecl *Func) {
  FunctionSignature Signature;

  for (const auto *Param : Func->parameters()) {
    StringRef Name = Param->getName();
    DeclarationFragments Fragments = getFragmentsForParam(Param);

    Signature.addParameter(Name, Fragments);
  }

  DeclarationFragments After;
  DeclarationFragments Returns =
      getFragmentsForType(Func->getReturnType(), Func->getASTContext(), After)
          .append(std::move(After));

  Signature.setReturnType(Returns);

  return Signature;
}

// Subheading of a symbol defaults to its name.
DeclarationFragments
DeclarationFragmentsBuilder::getSubHeading(const NamedDecl *Decl) {
  DeclarationFragments Fragments;
  if (!Decl->getName().empty())
    Fragments.append(Decl->getName(),
                     DeclarationFragments::FragmentKind::Identifier);
  return Fragments;
}

} // namespace symbolgraph
} // namespace clang
