//===-- include/flang/Semantics/scope.h -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_SCOPE_H_
#define FORTRAN_SEMANTICS_SCOPE_H_

#include "attr.h"
#include "symbol.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Common/reference.h"
#include "flang/Parser/message.h"
#include "flang/Parser/provenance.h"
#include <list>
#include <map>
#include <optional>
#include <set>
#include <string>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {

using namespace parser::literals;

using common::ConstantSubscript;

class SemanticsContext;

// An equivalence object is represented by a symbol for the variable name,
// the indices for an array element, and the lower bound for a substring.
struct EquivalenceObject {
  EquivalenceObject(Symbol &symbol, std::vector<ConstantSubscript> subscripts,
      std::optional<ConstantSubscript> substringStart, parser::CharBlock source)
      : symbol{symbol}, subscripts{subscripts},
        substringStart{substringStart}, source{source} {}

  bool operator==(const EquivalenceObject &) const;
  bool operator<(const EquivalenceObject &) const;
  std::string AsFortran() const;

  Symbol &symbol;
  std::vector<ConstantSubscript> subscripts; // for array elem
  std::optional<ConstantSubscript> substringStart;
  parser::CharBlock source;
};
using EquivalenceSet = std::vector<EquivalenceObject>;

class Scope {
  using mapType = std::map<SourceName, MutableSymbolRef>;

public:
  ENUM_CLASS(Kind, Global, Module, MainProgram, Subprogram, BlockData,
      DerivedType, Block, Forall, ImpliedDos)
  using ImportKind = common::ImportKind;

  // Create the Global scope -- the root of the scope tree
  Scope() : Scope{*this, Kind::Global, nullptr} {}
  Scope(Scope &parent, Kind kind, Symbol *symbol)
      : parent_{parent}, kind_{kind}, symbol_{symbol} {
    if (symbol) {
      symbol->set_scope(this);
    }
  }
  Scope(const Scope &) = delete;

  bool operator==(const Scope &that) const { return this == &that; }
  bool operator!=(const Scope &that) const { return this != &that; }

  Scope &parent() {
    CHECK(&parent_ != this);
    return parent_;
  }
  const Scope &parent() const {
    CHECK(&parent_ != this);
    return parent_;
  }
  Kind kind() const { return kind_; }
  bool IsGlobal() const { return kind_ == Kind::Global; }
  bool IsModule() const {
    return kind_ == Kind::Module &&
        !symbol_->get<ModuleDetails>().isSubmodule();
  }
  bool IsSubmodule() const {
    return kind_ == Kind::Module && symbol_->get<ModuleDetails>().isSubmodule();
  }
  bool IsDerivedType() const { return kind_ == Kind::DerivedType; }
  bool IsStmtFunction() const;
  bool IsParameterizedDerivedType() const;
  Symbol *symbol() { return symbol_; }
  const Symbol *symbol() const { return symbol_; }

  inline const Symbol *GetSymbol() const;
  const Scope *GetDerivedTypeParent() const;
  const Scope &GetDerivedTypeBase() const;
  inline std::optional<SourceName> GetName() const;
  bool Contains(const Scope &) const;
  /// Make a scope nested in this one
  Scope &MakeScope(Kind kind, Symbol *symbol = nullptr);

  using size_type = mapType::size_type;
  using iterator = mapType::iterator;
  using const_iterator = mapType::const_iterator;

  iterator begin() { return symbols_.begin(); }
  iterator end() { return symbols_.end(); }
  const_iterator begin() const { return symbols_.begin(); }
  const_iterator end() const { return symbols_.end(); }
  const_iterator cbegin() const { return symbols_.cbegin(); }
  const_iterator cend() const { return symbols_.cend(); }

  // Return symbols in declaration order (the iterators above are in name order)
  SymbolVector GetSymbols() const;
  MutableSymbolVector GetSymbols();

  iterator find(const SourceName &name);
  const_iterator find(const SourceName &name) const {
    return symbols_.find(name);
  }
  size_type erase(const SourceName &);
  bool empty() const { return symbols_.empty(); }

  // Look for symbol by name in this scope and host (depending on imports).
  Symbol *FindSymbol(const SourceName &) const;

  // Look for component symbol by name in a derived type's scope and
  // parents'.
  Symbol *FindComponent(SourceName) const;

  /// Make a Symbol with unknown details.
  std::pair<iterator, bool> try_emplace(
      const SourceName &name, Attrs attrs = Attrs()) {
    return try_emplace(name, attrs, UnknownDetails());
  }
  /// Make a Symbol with provided details.
  template <typename D>
  common::IfNoLvalue<std::pair<iterator, bool>, D> try_emplace(
      const SourceName &name, D &&details) {
    return try_emplace(name, Attrs(), std::move(details));
  }
  /// Make a Symbol with attrs and details
  template <typename D>
  common::IfNoLvalue<std::pair<iterator, bool>, D> try_emplace(
      const SourceName &name, Attrs attrs, D &&details) {
    Symbol &symbol{MakeSymbol(name, attrs, std::move(details))};
    return symbols_.emplace(name, symbol);
  }
  // Make a copy of a symbol in this scope; nullptr if one is already there
  Symbol *CopySymbol(const Symbol &);

  std::list<EquivalenceSet> &equivalenceSets() { return equivalenceSets_; }
  const std::list<EquivalenceSet> &equivalenceSets() const {
    return equivalenceSets_;
  }
  void add_equivalenceSet(EquivalenceSet &&);
  // Cray pointers are saved as map of pointee name -> pointer symbol
  const mapType &crayPointers() const { return crayPointers_; }
  void add_crayPointer(const SourceName &, Symbol &);
  mapType &commonBlocks() { return commonBlocks_; }
  const mapType &commonBlocks() const { return commonBlocks_; }
  Symbol &MakeCommonBlock(const SourceName &);
  Symbol *FindCommonBlock(const SourceName &);

  /// Make a Symbol but don't add it to the scope.
  template <typename D>
  common::IfNoLvalue<Symbol &, D> MakeSymbol(
      const SourceName &name, Attrs attrs, D &&details) {
    return allSymbols.Make(*this, name, attrs, std::move(details));
  }

  std::list<Scope> &children() { return children_; }
  const std::list<Scope> &children() const { return children_; }

  // For Module scope, maintain a mapping of all submodule scopes with this
  // module as its ancestor module. AddSubmodule returns false if already there.
  Scope *FindSubmodule(const SourceName &) const;
  bool AddSubmodule(const SourceName &, Scope &);

  const DeclTypeSpec *FindType(const DeclTypeSpec &) const;
  const DeclTypeSpec &MakeNumericType(TypeCategory, KindExpr &&kind);
  const DeclTypeSpec &MakeLogicalType(KindExpr &&kind);
  const DeclTypeSpec &MakeCharacterType(
      ParamValue &&length, KindExpr &&kind = KindExpr{0});
  DeclTypeSpec &MakeDerivedType(DeclTypeSpec::Category, DerivedTypeSpec &&);
  const DeclTypeSpec &MakeTypeStarType();
  const DeclTypeSpec &MakeClassStarType();

  std::size_t size() const { return size_; }
  void set_size(std::size_t size) { size_ = size; }
  std::size_t alignment() const { return alignment_; }
  void set_alignment(std::size_t alignment) { alignment_ = alignment; }

  ImportKind GetImportKind() const;
  // Names appearing in IMPORT statements in this scope
  std::set<SourceName> importNames() const { return importNames_; }

  // Set the kind of imports from host into this scope.
  // Return an error message for incompatible kinds.
  std::optional<parser::MessageFixedText> SetImportKind(ImportKind);

  void add_importName(const SourceName &);

  const DerivedTypeSpec *derivedTypeSpec() const { return derivedTypeSpec_; }
  DerivedTypeSpec *derivedTypeSpec() { return derivedTypeSpec_; }
  void set_derivedTypeSpec(DerivedTypeSpec &spec) { derivedTypeSpec_ = &spec; }

  bool hasSAVE() const { return hasSAVE_; }
  void set_hasSAVE(bool yes = true) { hasSAVE_ = yes; }

  // The range of the source of this and nested scopes.
  const parser::CharBlock &sourceRange() const { return sourceRange_; }
  void AddSourceRange(const parser::CharBlock &);
  // Find the smallest scope under this one that contains source
  const Scope *FindScope(parser::CharBlock) const;
  Scope *FindScope(parser::CharBlock);

  // Attempts to find a match for a derived type instance
  const DeclTypeSpec *FindInstantiatedDerivedType(const DerivedTypeSpec &,
      DeclTypeSpec::Category = DeclTypeSpec::TypeDerived) const;

  bool IsModuleFile() const {
    return kind_ == Kind::Module && symbol_ &&
        symbol_->test(Symbol::Flag::ModFile);
  }

  void InstantiateDerivedTypes(SemanticsContext &);

private:
  Scope &parent_; // this is enclosing scope, not extended derived type base
  const Kind kind_;
  std::size_t size_{0}; // size in bytes
  std::size_t alignment_{0}; // required alignment in bytes
  parser::CharBlock sourceRange_;
  Symbol *const symbol_; // if not null, symbol_->scope() == this
  std::list<Scope> children_;
  mapType symbols_;
  mapType commonBlocks_;
  std::list<EquivalenceSet> equivalenceSets_;
  mapType crayPointers_;
  std::map<SourceName, common::Reference<Scope>> submodules_;
  std::list<DeclTypeSpec> declTypeSpecs_;
  std::optional<ImportKind> importKind_;
  std::set<SourceName> importNames_;
  DerivedTypeSpec *derivedTypeSpec_{nullptr}; // dTS->scope() == this
  bool hasSAVE_{false}; // scope has a bare SAVE statement
  // When additional data members are added to Scope, remember to
  // copy them, if appropriate, in InstantiateDerivedType().

  // Storage for all Symbols. Every Symbol is in allSymbols and every Symbol*
  // or Symbol& points to one in there.
  static Symbols<1024> allSymbols;

  bool CanImport(const SourceName &) const;
  const DeclTypeSpec &MakeLengthlessType(DeclTypeSpec &&);

  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Scope &);
};

// Inline so that it can be called from Evaluate without a link-time dependency.

inline const Symbol *Scope::GetSymbol() const {
  return symbol_         ? symbol_
      : derivedTypeSpec_ ? &derivedTypeSpec_->typeSymbol()
                         : nullptr;
}

inline std::optional<SourceName> Scope::GetName() const {
  if (const auto *sym{GetSymbol()}) {
    return sym->name();
  } else {
    return std::nullopt;
  }
}

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_SCOPE_H_
