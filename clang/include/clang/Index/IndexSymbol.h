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
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DataTypes.h"

namespace clang {
  class Decl;
  class LangOptions;

namespace index {

enum class SymbolKind : uint8_t {
  Unknown,

  Module,
  Namespace,
  NamespaceAlias,
  Macro,

  Enum,
  Struct,
  Class,
  Protocol,
  Extension,
  Union,
  TypeAlias,

  Function,
  Variable,
  Field,
  EnumConstant,

  InstanceMethod,
  ClassMethod,
  StaticMethod,
  InstanceProperty,
  ClassProperty,
  StaticProperty,

  Constructor,
  Destructor,
  ConversionFunction,

  Parameter,
};

enum class SymbolLanguage {
  C,
  ObjC,
  CXX,
  Swift,
};

/// Language specific sub-kinds.
enum class SymbolSubKind {
  None,
  CXXCopyConstructor,
  CXXMoveConstructor,
  AccessorGetter,
  AccessorSetter,
};

/// Set of properties that provide additional info about a symbol.
enum class SymbolProperty : uint8_t {
  Generic                       = 1 << 0,
  TemplatePartialSpecialization = 1 << 1,
  TemplateSpecialization        = 1 << 2,
  UnitTest                      = 1 << 3,
  IBAnnotated                   = 1 << 4,
  IBOutletCollection            = 1 << 5,
  GKInspectable                 = 1 << 6,
  Local                         = 1 << 7,
};
static const unsigned SymbolPropertyBitNum = 8;
typedef unsigned SymbolPropertySet;

/// Set of roles that are attributed to symbol occurrences.
enum class SymbolRole : uint32_t {
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
  RelationCalledBy    = 1 << 13,
  RelationExtendedBy  = 1 << 14,
  RelationAccessorOf  = 1 << 15,
  RelationContainedBy = 1 << 16,
  RelationIBTypeOf    = 1 << 17,
  RelationSpecializationOf = 1 << 18,
};
static const unsigned SymbolRoleBitNum = 19;
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
  SymbolSubKind SubKind;
  SymbolPropertySet Properties;
  SymbolLanguage Lang;
};

SymbolInfo getSymbolInfo(const Decl *D);

bool isFunctionLocalSymbol(const Decl *D);

void applyForEachSymbolRole(SymbolRoleSet Roles,
                            llvm::function_ref<void(SymbolRole)> Fn);
bool applyForEachSymbolRoleInterruptible(SymbolRoleSet Roles,
                            llvm::function_ref<bool(SymbolRole)> Fn);
void printSymbolRoles(SymbolRoleSet Roles, raw_ostream &OS);

/// \returns true if no name was printed, false otherwise.
bool printSymbolName(const Decl *D, const LangOptions &LO, raw_ostream &OS);

StringRef getSymbolKindString(SymbolKind K);
StringRef getSymbolSubKindString(SymbolSubKind K);
StringRef getSymbolLanguageString(SymbolLanguage K);

void applyForEachSymbolProperty(SymbolPropertySet Props,
                            llvm::function_ref<void(SymbolProperty)> Fn);
void printSymbolProperties(SymbolPropertySet Props, raw_ostream &OS);

} // namespace index
} // namespace clang

#endif
