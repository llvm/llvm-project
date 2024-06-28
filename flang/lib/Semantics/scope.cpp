//===-- lib/Semantics/scope.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/scope.h"
#include "flang/Parser/characters.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

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
  std::string buf;
  llvm::raw_string_ostream ss{buf};
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

Scope &Scope::MakeScope(Kind kind, Symbol *symbol) {
  return children_.emplace_back(*this, kind, symbol, context_);
}

template <typename T>
static std::vector<common::Reference<T>> GetSortedSymbols(
    const std::map<SourceName, MutableSymbolRef> &symbols) {
  std::vector<common::Reference<T>> result;
  result.reserve(symbols.size());
  for (auto &pair : symbols) {
    result.push_back(*pair.second);
  }
  std::sort(result.begin(), result.end(), SymbolSourcePositionCompare{});
  return result;
}

MutableSymbolVector Scope::GetSymbols() {
  return GetSortedSymbols<Symbol>(symbols_);
}
SymbolVector Scope::GetSymbols() const {
  return GetSortedSymbols<const Symbol>(symbols_);
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
  } else if (IsSubmodule()) {
    const Scope *parent{symbol_->get<ModuleDetails>().parent()};
    return parent ? parent->FindSymbol(name) : nullptr;
  } else if (CanImport(name)) {
    return parent_->FindSymbol(name);
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
    return nullptr; // already exists
  } else {
    Symbol &result{*pair.first->second};
    result.flags() = symbol.flags();
    result.set_details(common::Clone(symbol.details()));
    return &result;
  }
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
Symbol *Scope::FindCommonBlock(const SourceName &name) const {
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

const DeclTypeSpec *Scope::GetType(const SomeExpr &expr) {
  if (auto dyType{expr.GetType()}) {
    if (dyType->IsAssumedType()) {
      return &MakeTypeStarType();
    } else if (dyType->IsUnlimitedPolymorphic()) {
      return &MakeClassStarType();
    } else {
      switch (dyType->category()) {
      case TypeCategory::Integer:
      case TypeCategory::Real:
      case TypeCategory::Complex:
        return &MakeNumericType(dyType->category(), KindExpr{dyType->kind()});
      case TypeCategory::Character:
        if (const ParamValue * lenParam{dyType->charLengthParamValue()}) {
          return &MakeCharacterType(
              ParamValue{*lenParam}, KindExpr{dyType->kind()});
        } else {
          auto lenExpr{dyType->GetCharLength()};
          if (!lenExpr) {
            lenExpr =
                std::get<evaluate::Expr<evaluate::SomeCharacter>>(expr.u).LEN();
          }
          if (lenExpr) {
            return &MakeCharacterType(
                ParamValue{SomeIntExpr{std::move(*lenExpr)},
                    common::TypeParamAttr::Len},
                KindExpr{dyType->kind()});
          }
        }
        break;
      case TypeCategory::Logical:
        return &MakeLogicalType(KindExpr{dyType->kind()});
      case TypeCategory::Derived:
        return &MakeDerivedType(dyType->IsPolymorphic()
                ? DeclTypeSpec::ClassDerived
                : DeclTypeSpec::TypeDerived,
            DerivedTypeSpec{dyType->GetDerivedTypeSpec()});
      }
    }
  }
  return nullptr;
}

Scope::ImportKind Scope::GetImportKind() const {
  if (importKind_) {
    return *importKind_;
  }
  if (symbol_ && !symbol_->attrs().test(Attr::MODULE)) {
    if (auto *details{symbol_->detailsIf<SubprogramDetails>()}) {
      if (details->isInterface()) {
        return ImportKind::None; // default for non-mod-proc interface body
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
      (kind != ImportKind::Only && *importKind_ != ImportKind::Only)) {
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
  if (IsTopLevel() || parent_->IsTopLevel()) {
    return false;
  }
  switch (GetImportKind()) {
    SWITCH_COVERS_ALL_CASES
  case ImportKind::None:
    return false;
  case ImportKind::All:
  case ImportKind::Default:
    return true;
  case ImportKind::Only:
    return importNames_.count(name) > 0;
  }
}

void Scope::AddSourceRange(parser::CharBlock source) {
  if (source.empty()) {
    return;
  }
  const parser::AllCookedSources &allCookedSources{context_.allCookedSources()};
  const parser::CookedSource *cooked{allCookedSources.Find(source)};
  if (!cooked) {
    CHECK(context_.IsTempName(source.ToString()));
    return;
  }
  for (auto *scope{this}; !scope->IsTopLevel(); scope = &scope->parent()) {
    CHECK(scope->sourceRange_.empty() == (scope->cookedSource_ == nullptr));
    if (!scope->cookedSource_) {
      context_.UpdateScopeIndex(*scope, source);
      scope->cookedSource_ = cooked;
      scope->sourceRange_ = source;
    } else if (scope->cookedSource_ == cooked) {
      auto combined{scope->sourceRange()};
      combined.ExtendToCover(source);
      context_.UpdateScopeIndex(*scope, combined);
      scope->sourceRange_ = combined;
    } else {
      // There's a bug that will be hard to fix; crash informatively
      const parser::AllSources &allSources{allCookedSources.allSources()};
      const auto describe{[&](parser::CharBlock src) {
        if (auto range{allCookedSources.GetProvenanceRange(src)}) {
          std::size_t offset;
          if (const parser::SourceFile *
              file{allSources.GetSourceFile(range->start(), &offset)}) {
            return "'"s + file->path() + "' at " + std::to_string(offset) +
                " for " + std::to_string(range->size());
          } else {
            return "(GetSourceFile failed)"s;
          }
        } else {
          return "(GetProvenanceRange failed)"s;
        }
      }};
      std::string scopeDesc{describe(scope->sourceRange_)};
      std::string newDesc{describe(source)};
      common::die("AddSourceRange would have combined ranges from distinct "
                  "source files \"%s\" and \"%s\"",
          scopeDesc.c_str(), newDesc.c_str());
    }
    // Note: If the "break;" here were unconditional (or, equivalently, if
    // there were no loop at all) then the source ranges of parent scopes
    // would not enclose the source ranges of their children.  Timing
    // shows that it's cheap to maintain this property, with the exceptions
    // of top-level scopes and for (sub)modules and their descendant
    // submodules.
    if (scope->IsSubmodule()) {
      break; // Submodules are child scopes but not contained ranges
    }
  }
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Scope &scope) {
  os << Scope::EnumToString(scope.kind()) << " scope: ";
  if (auto *symbol{scope.symbol()}) {
    os << *symbol << ' ';
  }
  if (scope.derivedTypeSpec_) {
    os << "instantiation of " << *scope.derivedTypeSpec_ << ' ';
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

bool Scope::IsStmtFunction() const {
  return symbol_ && symbol_->test(Symbol::Flag::StmtFunction);
}

template <common::TypeParamAttr... ParamAttr> struct IsTypeParamHelper {
  static_assert(sizeof...(ParamAttr) == 0, "must have one or zero template");
  static bool IsParam(const Symbol &symbol) {
    return symbol.has<TypeParamDetails>();
  }
};

template <common::TypeParamAttr ParamAttr> struct IsTypeParamHelper<ParamAttr> {
  static bool IsParam(const Symbol &symbol) {
    if (const auto *typeParam{symbol.detailsIf<TypeParamDetails>()}) {
      return typeParam->attr() == ParamAttr;
    }
    return false;
  }
};

template <common::TypeParamAttr... ParamAttr>
static bool IsParameterizedDerivedTypeHelper(const Scope &scope) {
  if (scope.IsDerivedType()) {
    if (const Scope * parent{scope.GetDerivedTypeParent()}) {
      if (IsParameterizedDerivedTypeHelper<ParamAttr...>(*parent)) {
        return true;
      }
    }
    for (const auto &nameAndSymbolPair : scope) {
      if (IsTypeParamHelper<ParamAttr...>::IsParam(*nameAndSymbolPair.second)) {
        return true;
      }
    }
  }
  return false;
}

bool Scope::IsParameterizedDerivedType() const {
  return IsParameterizedDerivedTypeHelper<>(*this);
}
bool Scope::IsDerivedTypeWithLengthParameter() const {
  return IsParameterizedDerivedTypeHelper<common::TypeParamAttr::Len>(*this);
}
bool Scope::IsDerivedTypeWithKindParameter() const {
  return IsParameterizedDerivedTypeHelper<common::TypeParamAttr::Kind>(*this);
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

const Scope *Scope::GetDerivedTypeParent() const {
  if (const Symbol * symbol{GetSymbol()}) {
    if (const DerivedTypeSpec * parent{symbol->GetParentTypeSpec(this)}) {
      return parent->scope();
    }
  }
  return nullptr;
}

const Scope &Scope::GetDerivedTypeBase() const {
  const Scope *child{this};
  for (const Scope *parent{GetDerivedTypeParent()}; parent != nullptr;
       parent = child->GetDerivedTypeParent()) {
    child = parent;
  }
  return *child;
}

void Scope::InstantiateDerivedTypes() {
  for (DeclTypeSpec &type : declTypeSpecs_) {
    if (type.category() == DeclTypeSpec::TypeDerived ||
        type.category() == DeclTypeSpec::ClassDerived) {
      type.derivedTypeSpec().Instantiate(*this);
    }
  }
}
} // namespace Fortran::semantics
