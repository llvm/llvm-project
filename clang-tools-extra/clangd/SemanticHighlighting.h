//==-- SemanticHighlighting.h - Generating highlights from the AST-- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file supports semantic highlighting: categorizing tokens in the file so
// that the editor can color/style them differently.
// This is particularly valuable for C++: its complex and context-dependent
// grammar is a challenge for simple syntax-highlighting techniques.
//
// Semantic highlightings are calculated for an AST by visiting every AST node
// and classifying nodes that are interesting to highlight (variables/function
// calls etc.).
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANGD_SEMANTICHIGHLIGHTING_H

#include "Protocol.h"

#include "clang/AST/TypeBase.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
class NamedDecl;

namespace clangd {
class ParsedAST;

enum class HighlightingKind {
  Variable = 0,
  LocalVariable,
  Parameter,
  Function,
  Method,
  StaticMethod,
  Field,
  StaticField,
  Class,
  Interface,
  Enum,
  EnumConstant,
  Typedef,
  Type,
  Unknown,
  Namespace,
  TemplateParameter,
  Concept,
  Primitive,
  Macro,
  Modifier,
  Operator,
  Bracket,
  Label,

  // This one is different from the other kinds as it's a line style
  // rather than a token style.
  InactiveCode,

  LastKind = InactiveCode
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, HighlightingKind K);
std::optional<HighlightingKind>
highlightingKindFromString(llvm::StringRef Name);

enum class HighlightingModifier {
  Declaration,
  Definition,
  Deprecated,
  Deduced,
  Readonly,
  Static,
  Abstract,
  Virtual,
  DependentName,
  DefaultLibrary,
  UsedAsMutableReference,
  UsedAsMutablePointer,
  ConstructorOrDestructor,
  UserDefined,

  FunctionScope,
  ClassScope,
  FileScope,
  GlobalScope,

  LastModifier = GlobalScope
};
static_assert(static_cast<unsigned>(HighlightingModifier::LastModifier) < 32,
              "Increase width of modifiers bitfield!");
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, HighlightingModifier K);
std::optional<HighlightingModifier>
highlightingModifierFromString(llvm::StringRef Name);

// Contains all information needed for the highlighting a token.
struct HighlightingToken {
  HighlightingKind Kind;
  uint32_t Modifiers = 0;
  Range R;

  HighlightingToken &addModifier(HighlightingModifier M) {
    Modifiers |= 1 << static_cast<unsigned>(M);
    return *this;
  }
};

bool operator==(const HighlightingToken &L, const HighlightingToken &R);
bool operator<(const HighlightingToken &L, const HighlightingToken &R);

// Returns all HighlightingTokens from an AST. Only generates highlights for the
// main AST.
std::vector<HighlightingToken>
getSemanticHighlightings(ParsedAST &AST, bool IncludeInactiveRegionTokens);

std::vector<SemanticToken> toSemanticTokens(llvm::ArrayRef<HighlightingToken>,
                                            llvm::StringRef Code);
llvm::StringRef toSemanticTokenType(HighlightingKind Kind);
llvm::StringRef toSemanticTokenModifier(HighlightingModifier Modifier);
std::vector<SemanticTokensEdit> diffTokens(llvm::ArrayRef<SemanticToken> Before,
                                           llvm::ArrayRef<SemanticToken> After);

// Returns ranges of the file that are inside an inactive preprocessor branch.
// The preprocessor directives at the beginning and end of a branch themselves
// are not included.
std::vector<Range> getInactiveRegions(ParsedAST &AST);


// Whether T is const in a loose sense - is a variable with this type readonly?
bool isConst(QualType T);

// Whether D is const in a loose sense (should it be highlighted as such?)
// FIXME: This is separate from whether *a particular usage* can mutate D.
//        We may want V in V.size() to be readonly even if V is mutable.
bool isConst(const Decl *D);

// "Static" means many things in C++, only some get the "static" modifier.
//
// Meanings that do:
// - Members associated with the class rather than the instance.
//   This is what 'static' most often means across languages.
// - static local variables
//   These are similarly "detached from their context" by the static keyword.
//   In practice, these are rarely used inside classes, reducing confusion.
//
// Meanings that don't:
// - Namespace-scoped variables, which have static storage class.
//   This is implicit, so the keyword "static" isn't so strongly associated.
//   If we want a modifier for these, "global scope" is probably the concept.
// - Namespace-scoped variables/functions explicitly marked "static".
//   There the keyword changes *linkage* , which is a totally different concept.
//   If we want to model this, "file scope" would be a nice modifier.
//
// This is confusing, and maybe we should use another name, but because "static"
// is a standard LSP modifier, having one with that name has advantages.
bool isStatic(const Decl *D);
// Indicates whether declaration D is abstract in cases where D is a struct or a
// class.
bool isAbstract(const Decl *D);
// Indicates whether declaration D is virtual in cases where D is a method.
bool isVirtual(const Decl *D);
// Indicates whether declaration D is final in cases where D is a struct, class
// or method.
bool isFinal(const Decl *D);
// Indicates whether declaration D is a unique definition (as opposed to a
// declaration).
bool isUniqueDefinition(const NamedDecl *Decl);

} // namespace clangd
} // namespace clang

#endif
