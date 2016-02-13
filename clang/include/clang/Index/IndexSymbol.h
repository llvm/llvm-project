//===--- IndexSymbol.h - Types and functions for indexing symbols ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_INDEX_INDEXSYMBOL_H
#define LLVM_CLANG_INDEX_INDEXSYMBOL_H

#include "clang/Basic/LLVM.h"
#include "llvm/Support/DataTypes.h"

namespace clang {
  class Decl;

namespace index {

enum class SymbolKind : uint8_t {
  Unknown,

  Module,
  Macro,

  Enum,
  Struct,
  Union,
  Typedef,

  Function,
  Variable,
  Field,
  EnumConstant,

  ObjCClass,
  ObjCProtocol,
  ObjCCategory,

  ObjCInstanceMethod,
  ObjCClassMethod,
  ObjCProperty,
  ObjCIvar,

  CXXClass,
  CXXNamespace,
  CXXNamespaceAlias,
  CXXStaticVariable,
  CXXStaticMethod,
  CXXInstanceMethod,
  CXXConstructor,
  CXXDestructor,
  CXXConversionFunction,
  CXXTypeAlias,
  CXXInterface,
};

enum class SymbolLanguage {
  C,
  ObjC,
  CXX,
};

enum class SymbolCXXTemplateKind {
  NonTemplate,
  Template,
  TemplatePartialSpecialization,
  TemplateSpecialization,
};

/// Set of roles that are attributed to symbol occurrences.
enum class SymbolRole : uint16_t {
  Declaration = 1 << 0,
  Definition  = 1 << 1,
  Reference   = 1 << 2,
  Read        = 1 << 3,
  Write       = 1 << 4,
  Call        = 1 << 5,
  Dynamic     = 1 << 6,
  AddressOf   = 1 << 7,
  Implicit    = 1 << 8,

  // Relation roles.
  RelationChildOf     = 1 << 9,
  RelationBaseOf      = 1 << 10,
  RelationOverrideOf  = 1 << 11,
  RelationReceivedBy  = 1 << 12,
};
static const unsigned SymbolRoleBitNum = 13;
typedef unsigned SymbolRoleSet;

/// Represents a relation to another symbol for a symbol occurrence.
struct SymbolRelation {
  SymbolRoleSet Roles;
  const Decl *RelatedSymbol;

  SymbolRelation(SymbolRoleSet Roles, const Decl *Sym)
    : Roles(Roles), RelatedSymbol(Sym) {}
};

struct SymbolInfo {
  SymbolKind Kind;
  SymbolCXXTemplateKind TemplateKind;
  SymbolLanguage Lang;
};

SymbolInfo getSymbolInfo(const Decl *D);

} // namespace index
} // namespace clang

#endif
