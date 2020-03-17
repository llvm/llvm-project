//===-- lib/Semantics/scope.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/scope.h"
#include "flang/Parser/characters.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/type.h"
#include <algorithm>
#include <memory>
#include <sstream>

namespace Fortran::semantics {

Symbols<1024> Scope::allSymbols;

bool EquivalenceObject::operator==(const EquivalenceObject &that) const {
  return symbol == that.symbol && subscripts == that.subscripts &&
      substringStart == that.substringStart;
}

bool EquivalenceObject::operator<(const EquivalenceObject &that) const {
  return &symbol < &that.symbol ||
      (&symbol == &that.symbol &&
          (subscripts < that.subscripts ||
              (subscripts == that.subscripts &&
                  substringStart < that.substringStart)));
}

std::string EquivalenceObject::AsFortran() const {
  std::stringstream ss;
  ss << symbol.name().ToString();
  if (!subscripts.empty()) {
    char sep{'('};
    for (auto subscript : subscripts) {
      ss << sep << subscript;
      sep = ',';
    }
    ss << ')';
  }
  if (substringStart) {
    ss << '(' << *substringStart << ":)";
  }
  return ss.str();
}

bool Scope::IsModule() const {
  return kind_ == Kind::Module && !symbol_->get<ModuleDetails>().isSubmodule();
}
bool Scope::IsSubmodule() const {
  return kind_ == Kind::Module && symbol_->get<ModuleDetails>().isSubmodule();
}

Scope &Scope::MakeScope(Kind kind, Symbol *symbol) {
  return children_.emplace_back(*this, kind, symbol);
}

Scope::iterator Scope::find(const SourceName &name) {
  return symbols_.find(name);
}
Scope::size_type Scope::erase(const SourceName &name) {
  auto it{symbols_.find(name)};
  if (it != end()) {
    symbols_.erase(it);
    return 1;
  } else {
    return 0;
  }
}
Symbol *Scope::FindSymbol(const SourceName &name) const {
  auto it{find(name)};
  if (it != end()) {
    return &*it->second;
  } else if (CanImport(name)) {
    return parent_.FindSymbol(name);
  } else {
    return nullptr;
  }
}

Symbol *Scope::FindComponent(SourceName name) const {
  CHECK(IsDerivedType());
  auto found{find(name)};
  if (found != end()) {
    return &*found->second;
  } else if (const Scope * parent{GetDerivedTypeParent()}) {
    return parent->FindComponent(name);
  } else {
    return nullptr;
  }
}

std::optional<SourceName> Scope::GetName() const {
  if (const auto *sym{GetSymbol()}) {
    return sym->name();
  } else {
    return std::nullopt;
  }
}

bool Scope::Contains(const Scope &that) const {
  for (const Scope *scope{&that};; scope = &scope->parent()) {
    if (*scope == *this) {
      return true;
    }
    if (scope->IsGlobal()) {
      return false;
    }
  }
}

Symbol *Scope::CopySymbol(const Symbol &symbol) {
  auto pair{try_emplace(symbol.name(), symbol.attrs())};
  if (!pair.second) {
    return nullptr;  // already exists
  } else {
    Symbol &result{*pair.first->second};
    result.flags() = symbol.flags();
    result.set_details(common::Clone(symbol.details()));
    return &result;
  }
}

const std::list<EquivalenceSet> &Scope::equivalenceSets() const {
  return equivalenceSets_;
}
void Scope::add_equivalenceSet(EquivalenceSet &&set) {
  equivalenceSets_.emplace_back(std::move(set));
}

void Scope::add_crayPointer(const SourceName &name, Symbol &pointer) {
  CHECK(pointer.test(Symbol::Flag::CrayPointer));
  crayPointers_.emplace(name, pointer);
}

Symbol &Scope::MakeCommonBlock(const SourceName &name) {
  const auto it{commonBlocks_.find(name)};
  if (it != commonBlocks_.end()) {
    return *it->second;
  } else {
    Symbol &symbol{MakeSymbol(name, Attrs{}, CommonBlockDetails{})};
    commonBlocks_.emplace(name, symbol);
    return symbol;
  }
}
Symbol *Scope::FindCommonBlock(const SourceName &name) {
  const auto it{commonBlocks_.find(name)};
  return it != commonBlocks_.end() ? &*it->second : nullptr;
}

Scope *Scope::FindSubmodule(const SourceName &name) const {
  auto it{submodules_.find(name)};
  if (it == submodules_.end()) {
    return nullptr;
  } else {
    return &*it->second;
  }
}
bool Scope::AddSubmodule(const SourceName &name, Scope &submodule) {
  return submodules_.emplace(name, submodule).second;
}

const DeclTypeSpec *Scope::FindType(const DeclTypeSpec &type) const {
  auto it{std::find(declTypeSpecs_.begin(), declTypeSpecs_.end(), type)};
  return it != declTypeSpecs_.end() ? &*it : nullptr;
}

const DeclTypeSpec &Scope::MakeNumericType(
    TypeCategory category, KindExpr &&kind) {
  return MakeLengthlessType(NumericTypeSpec{category, std::move(kind)});
}
const DeclTypeSpec &Scope::MakeLogicalType(KindExpr &&kind) {
  return MakeLengthlessType(LogicalTypeSpec{std::move(kind)});
}
const DeclTypeSpec &Scope::MakeTypeStarType() {
  return MakeLengthlessType(DeclTypeSpec{DeclTypeSpec::TypeStar});
}
const DeclTypeSpec &Scope::MakeClassStarType() {
  return MakeLengthlessType(DeclTypeSpec{DeclTypeSpec::ClassStar});
}
// Types that can't have length parameters can be reused without having to
// compare length expressions. They are stored in the global scope.
const DeclTypeSpec &Scope::MakeLengthlessType(DeclTypeSpec &&type) {
  const auto *found{FindType(type)};
  return found ? *found : declTypeSpecs_.emplace_back(std::move(type));
}

const DeclTypeSpec &Scope::MakeCharacterType(
    ParamValue &&length, KindExpr &&kind) {
  return declTypeSpecs_.emplace_back(
      CharacterTypeSpec{std::move(length), std::move(kind)});
}

DeclTypeSpec &Scope::MakeDerivedType(
    DeclTypeSpec::Category category, DerivedTypeSpec &&spec) {
  return declTypeSpecs_.emplace_back(category, std::move(spec));
}

void Scope::set_chars(parser::CookedSource &cooked) {
  CHECK(kind_ == Kind::Module);
  CHECK(parent_.IsGlobal() || parent_.IsModuleFile());
  CHECK(DEREF(symbol_).test(Symbol::Flag::ModFile));
  // TODO: Preserve the CookedSource rather than acquiring its string.
  chars_ = cooked.AcquireData();
}

Scope::ImportKind Scope::GetImportKind() const {
  if (importKind_) {
    return *importKind_;
  }
  if (symbol_ && !symbol_->attrs().test(Attr::MODULE)) {
    if (auto *details{symbol_->detailsIf<SubprogramDetails>()}) {
      if (details->isInterface()) {
        return ImportKind::None;  // default for non-mod-proc interface body
      }
    }
  }
  return ImportKind::Default;
}

std::optional<parser::MessageFixedText> Scope::SetImportKind(ImportKind kind) {
  if (!importKind_) {
    importKind_ = kind;
    return std::nullopt;
  }
  bool hasNone{kind == ImportKind::None || *importKind_ == ImportKind::None};
  bool hasAll{kind == ImportKind::All || *importKind_ == ImportKind::All};
  // Check C8100 and C898: constraints on multiple IMPORT statements
  if (hasNone || hasAll) {
    return hasNone
        ? "IMPORT,NONE must be the only IMPORT statement in a scope"_err_en_US
        : "IMPORT,ALL must be the only IMPORT statement in a scope"_err_en_US;
  } else if (kind != *importKind_ &&
      (kind != ImportKind::Only || kind != ImportKind::Only)) {
    return "Every IMPORT must have ONLY specifier if one of them does"_err_en_US;
  } else {
    return std::nullopt;
  }
}

void Scope::add_importName(const SourceName &name) {
  importNames_.insert(name);
}

// true if name can be imported or host-associated from parent scope.
bool Scope::CanImport(const SourceName &name) const {
  if (IsGlobal() || parent_.IsGlobal()) {
    return false;
  }
  switch (GetImportKind()) {
    SWITCH_COVERS_ALL_CASES
  case ImportKind::None: return false;
  case ImportKind::All:
  case ImportKind::Default: return true;
  case ImportKind::Only: return importNames_.count(name) > 0;
  }
}

const Scope *Scope::FindScope(parser::CharBlock source) const {
  return const_cast<Scope *>(this)->FindScope(source);
}

Scope *Scope::FindScope(parser::CharBlock source) {
  bool isContained{sourceRange_.Contains(source)};
  if (!isContained && !IsGlobal() && !IsModuleFile()) {
    return nullptr;
  }
  for (auto &child : children_) {
    if (auto *scope{child.FindScope(source)}) {
      return scope;
    }
  }
  return isContained ? this : nullptr;
}

void Scope::AddSourceRange(const parser::CharBlock &source) {
  for (auto *scope = this; !scope->IsGlobal(); scope = &scope->parent()) {
    scope->sourceRange_.ExtendToCover(source);
  }
}

std::ostream &operator<<(std::ostream &os, const Scope &scope) {
  os << Scope::EnumToString(scope.kind()) << " scope: ";
  if (auto *symbol{scope.symbol()}) {
    os << *symbol << ' ';
  }
  os << scope.children_.size() << " children\n";
  for (const auto &pair : scope.symbols_) {
    const Symbol &symbol{*pair.second};
    os << "  " << symbol << '\n';
  }
  if (!scope.equivalenceSets_.empty()) {
    os << "  Equivalence Sets:\n";
    for (const auto &set : scope.equivalenceSets_) {
      os << "   ";
      for (const auto &object : set) {
        os << ' ' << object.AsFortran();
      }
      os << '\n';
    }
  }
  for (const auto &pair : scope.commonBlocks_) {
    const Symbol &symbol{*pair.second};
    os << "  " << symbol << '\n';
  }
  return os;
}

bool Scope::IsParameterizedDerivedType() const {
  if (!IsDerivedType()) {
    return false;
  }
  if (const Scope * parent{GetDerivedTypeParent()}) {
    if (parent->IsParameterizedDerivedType()) {
      return true;
    }
  }
  for (const auto &pair : symbols_) {
    if (pair.second->has<TypeParamDetails>()) {
      return true;
    }
  }
  return false;
}

const DeclTypeSpec *Scope::FindInstantiatedDerivedType(
    const DerivedTypeSpec &spec, DeclTypeSpec::Category category) const {
  DeclTypeSpec type{category, spec};
  if (const auto *result{FindType(type)}) {
    return result;
  } else if (IsGlobal()) {
    return nullptr;
  } else {
    return parent().FindInstantiatedDerivedType(spec, category);
  }
}

const Symbol *Scope::GetSymbol() const {
  if (symbol_) {
    return symbol_;
  }
  if (derivedTypeSpec_) {
    return &derivedTypeSpec_->typeSymbol();
  }
  return nullptr;
}

const Scope *Scope::GetDerivedTypeParent() const {
  if (const Symbol * symbol{GetSymbol()}) {
    if (const DerivedTypeSpec * parent{symbol->GetParentTypeSpec(this)}) {
      return parent->scope();
    }
  }
  return nullptr;
}

void Scope::InstantiateDerivedTypes(SemanticsContext &context) {
  for (DeclTypeSpec &type : declTypeSpecs_) {
    if (type.category() == DeclTypeSpec::TypeDerived ||
        type.category() == DeclTypeSpec::ClassDerived) {
      type.derivedTypeSpec().Instantiate(*this, context);
    }
  }
}
}
