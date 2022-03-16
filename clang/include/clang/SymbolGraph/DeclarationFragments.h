//===- SymbolGraph/DeclarationFragments.h -----------------------*- C++ -*-===//
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

#ifndef LLVM_CLANG_SYMBOLGRAPH_DECLARATION_FRAGMENTS_H
#define LLVM_CLANG_SYMBOLGRAPH_DECLARATION_FRAGMENTS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "llvm/ADT/StringRef.h"
#include <vector>

namespace clang {
namespace symbolgraph {

class DeclarationFragments {
public:
  DeclarationFragments() = default;

  enum class FragmentKind {
    None,
    Keyword,
    Attribute,
    NumberLiteral,
    StringLiteral,
    Identifier,
    TypeIdentifier,
    GenericParameter,
    ExternalParam,
    InternalParam,
    Text,
  };

  struct Fragment {
    std::string Spelling;
    FragmentKind Kind;
    std::string PreciseIdentifier;

    Fragment(StringRef Spelling, FragmentKind Kind, StringRef PreciseIdentifier)
        : Spelling(Spelling), Kind(Kind), PreciseIdentifier(PreciseIdentifier) {
    }
  };

  const std::vector<Fragment> &getFragments() const { return Fragments; }

  DeclarationFragments &append(StringRef Spelling, FragmentKind Kind,
                               StringRef PreciseIdentifier = "") {
    if (Kind == FragmentKind::Text && !Fragments.empty() &&
        Fragments.back().Kind == FragmentKind::Text) {
      Fragments.back().Spelling.append(Spelling.data(), Spelling.size());
    } else {
      Fragments.emplace_back(Spelling, Kind, PreciseIdentifier);
    }
    return *this;
  }

  DeclarationFragments &append(DeclarationFragments &&Other) {
    Fragments.insert(Fragments.end(),
                     std::make_move_iterator(Other.Fragments.begin()),
                     std::make_move_iterator(Other.Fragments.end()));
    Other.Fragments.clear();
    return *this;
  }

  DeclarationFragments &appendSpace();

  static StringRef getFragmentKindString(FragmentKind Kind);
  static FragmentKind parseFragmentKindFromString(StringRef S);

private:
  std::vector<Fragment> Fragments;
};

class FunctionSignature {
public:
  FunctionSignature() = default;

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

  bool empty() const {
    return Parameters.empty() && ReturnType.getFragments().empty();
  }

private:
  std::vector<Parameter> Parameters;
  DeclarationFragments ReturnType;
};

class DeclarationFragmentsBuilder {
public:
  static DeclarationFragments getFragmentsForVar(const VarDecl *);
  static DeclarationFragments getFragmentsForFunction(const FunctionDecl *);
  static DeclarationFragments getSubHeading(const NamedDecl *);
  static FunctionSignature getFunctionSignature(const FunctionDecl *);

private:
  DeclarationFragmentsBuilder() = delete;

  static DeclarationFragments getFragmentsForType(const QualType, ASTContext &,
                                                  DeclarationFragments &);
  static DeclarationFragments getFragmentsForType(const Type *, ASTContext &,
                                                  DeclarationFragments &);
  static DeclarationFragments getFragmentsForNNS(const NestedNameSpecifier *,
                                                 ASTContext &,
                                                 DeclarationFragments &);
  static DeclarationFragments getFragmentsForQualifiers(const Qualifiers quals);
  static DeclarationFragments getFragmentsForParam(const ParmVarDecl *);
};

} // namespace symbolgraph
} // namespace clang

#endif // LLVM_CLANG_SYMBOLGRAPH_DECLARATION_FRAGMENTS_H
