//===- ExtractAPI/DeclarationFragments.h ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the Declaration Fragments related classes.
///
/// Declaration Fragments represent parts of a symbol declaration tagged with
/// syntactic/semantic information.
/// See https://github.com/apple/swift-docc-symbolkit
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_DECLARATION_FRAGMENTS_H
#define LLVM_CLANG_EXTRACTAPI_DECLARATION_FRAGMENTS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
namespace extractapi {

/// DeclarationFragments is a vector of tagged important parts of a symbol's
/// declaration.
///
/// The fragments sequence can be joined to form spans of declaration text, with
/// attached information useful for purposes like syntax-highlighting etc.
/// For example:
/// \code
///   const -> keyword    "const"
///   int   -> type       "int"
///   pi;   -> identifier "pi"
/// \endcode
class DeclarationFragments {
public:
  DeclarationFragments() = default;

  /// The kind of a fragment.
  enum class FragmentKind {
    /// Unknown fragment kind.
    None,

    Keyword,
    Attribute,
    NumberLiteral,
    StringLiteral,
    Identifier,

    /// Identifier that refers to a type in the context.
    TypeIdentifier,

    /// Parameter that's used as generics in the context. For example template
    /// parameters.
    GenericParameter,

    /// External parameters in Objective-C methods.
    /// For example, \c forKey in
    /// \code{.m}
    ///   - (void) setValue:(Value)value forKey(Key)key
    /// \endcode
    ExternalParam,

    /// Internal/local parameters in Objective-C methods.
    /// For example, \c key in
    /// \code{.m}
    ///   - (void) setValue:(Value)value forKey(Key)key
    /// \endcode
    InternalParam,

    Text,
  };

  /// Fragment holds information of a single fragment.
  struct Fragment {
    std::string Spelling;
    FragmentKind Kind;

    /// The USR of the fragment symbol, if applicable.
    std::string PreciseIdentifier;

    Fragment(StringRef Spelling, FragmentKind Kind, StringRef PreciseIdentifier)
        : Spelling(Spelling), Kind(Kind), PreciseIdentifier(PreciseIdentifier) {
    }
  };

  const std::vector<Fragment> &getFragments() const { return Fragments; }

  /// Append a new Fragment to the end of the Fragments.
  ///
  /// \returns a reference to the DeclarationFragments object itself after
  /// appending to chain up consecutive appends.
  DeclarationFragments &append(StringRef Spelling, FragmentKind Kind,
                               StringRef PreciseIdentifier = "") {
    if (Kind == FragmentKind::Text && !Fragments.empty() &&
        Fragments.back().Kind == FragmentKind::Text) {
      // If appending a text fragment, and the last fragment is also text,
      // merge into the last fragment.
      Fragments.back().Spelling.append(Spelling.data(), Spelling.size());
    } else {
      Fragments.emplace_back(Spelling, Kind, PreciseIdentifier);
    }
    return *this;
  }

  /// Append another DeclarationFragments to the end.
  ///
  /// Note: \p Other is moved from and cannot be used after a call to this
  /// method.
  ///
  /// \returns a reference to the DeclarationFragments object itself after
  /// appending to chain up consecutive appends.
  DeclarationFragments &append(DeclarationFragments &&Other) {
    Fragments.insert(Fragments.end(),
                     std::make_move_iterator(Other.Fragments.begin()),
                     std::make_move_iterator(Other.Fragments.end()));
    Other.Fragments.clear();
    return *this;
  }

  /// Append a text Fragment of a space character.
  ///
  /// \returns a reference to the DeclarationFragments object itself after
  /// appending to chain up consecutive appends.
  DeclarationFragments &appendSpace();

  /// Get the string description of a FragmentKind \p Kind.
  static StringRef getFragmentKindString(FragmentKind Kind);

  /// Get the corresponding FragmentKind from string \p S.
  static FragmentKind parseFragmentKindFromString(StringRef S);

private:
  std::vector<Fragment> Fragments;
};

/// Store function signature information with DeclarationFragments of the
/// return type and parameters.
class FunctionSignature {
public:
  FunctionSignature() = default;

  /// Parameter holds the name and DeclarationFragments of a single parameter.
  struct Parameter {
    std::string Name;
    DeclarationFragments Fragments;

    Parameter(StringRef Name, DeclarationFragments Fragments)
        : Name(Name), Fragments(Fragments) {}
  };

  const std::vector<Parameter> &getParameters() const { return Parameters; }
  const DeclarationFragments &getReturnType() const { return ReturnType; }

  FunctionSignature &addParameter(StringRef Name,
                                  DeclarationFragments Fragments) {
    Parameters.emplace_back(Name, Fragments);
    return *this;
  }

  void setReturnType(DeclarationFragments RT) { ReturnType = RT; }

  /// Determine if the FunctionSignature is empty.
  ///
  /// \returns true if the return type DeclarationFragments is empty and there
  /// is no parameter, otherwise false.
  bool empty() const {
    return Parameters.empty() && ReturnType.getFragments().empty();
  }

private:
  std::vector<Parameter> Parameters;
  DeclarationFragments ReturnType;
};

/// A factory class to build DeclarationFragments for different kinds of Decl.
class DeclarationFragmentsBuilder {
public:
  /// Build DeclarationFragments for a variable declaration VarDecl.
  static DeclarationFragments getFragmentsForVar(const VarDecl *);

  /// Build DeclarationFragments for a function declaration FunctionDecl.
  static DeclarationFragments getFragmentsForFunction(const FunctionDecl *);

  /// Build DeclarationFragments for an enum constant declaration
  /// EnumConstantDecl.
  static DeclarationFragments
  getFragmentsForEnumConstant(const EnumConstantDecl *);

  /// Build DeclarationFragments for an enum declaration EnumDecl.
  static DeclarationFragments getFragmentsForEnum(const EnumDecl *);

  /// Build DeclarationFragments for a field declaration FieldDecl.
  static DeclarationFragments getFragmentsForField(const FieldDecl *);

  /// Build DeclarationFragments for a struct record declaration RecordDecl.
  static DeclarationFragments getFragmentsForStruct(const RecordDecl *);

  /// Build sub-heading fragments for a NamedDecl.
  static DeclarationFragments getSubHeading(const NamedDecl *);

  /// Build FunctionSignature for a function declaration FunctionDecl.
  static FunctionSignature getFunctionSignature(const FunctionDecl *);

private:
  DeclarationFragmentsBuilder() = delete;

  /// Build DeclarationFragments for a QualType.
  static DeclarationFragments getFragmentsForType(const QualType, ASTContext &,
                                                  DeclarationFragments &);

  /// Build DeclarationFragments for a Type.
  static DeclarationFragments getFragmentsForType(const Type *, ASTContext &,
                                                  DeclarationFragments &);

  /// Build DeclarationFragments for a NestedNameSpecifier.
  static DeclarationFragments getFragmentsForNNS(const NestedNameSpecifier *,
                                                 ASTContext &,
                                                 DeclarationFragments &);

  /// Build DeclarationFragments for Qualifiers.
  static DeclarationFragments getFragmentsForQualifiers(const Qualifiers quals);

  /// Build DeclarationFragments for a parameter variable declaration
  /// ParmVarDecl.
  static DeclarationFragments getFragmentsForParam(const ParmVarDecl *);
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_DECLARATION_FRAGMENTS_H
