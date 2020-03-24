//===-- lib/Semantics/symbol.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/symbol.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace Fortran::semantics {

template<typename T>
static void DumpOptional(llvm::raw_ostream &os, const char *label, const T &x) {
  if (x) {
    os << ' ' << label << ':' << *x;
  }
}
template<typename T>
static void DumpExpr(llvm::raw_ostream &os, const char *label,
    const std::optional<evaluate::Expr<T>> &x) {
  if (x) {
    x->AsFortran(os << ' ' << label << ':');
  }
}

static void DumpBool(llvm::raw_ostream &os, const char *label, bool x) {
  if (x) {
    os << ' ' << label;
  }
}

static void DumpSymbolVector(llvm::raw_ostream &os, const SymbolVector &list) {
  char sep{' '};
  for (const Symbol &elem : list) {
    os << sep << elem.name();
    sep = ',';
  }
}

static void DumpType(llvm::raw_ostream &os, const Symbol &symbol) {
  if (const auto *type{symbol.GetType()}) {
    os << *type << ' ';
  }
}
static void DumpType(llvm::raw_ostream &os, const DeclTypeSpec *type) {
  if (type) {
    os << ' ' << *type;
  }
}

template<typename T>
static void DumpList(llvm::raw_ostream &os, const char *label, const T &list) {
  if (!list.empty()) {
    os << ' ' << label << ':';
    char sep{' '};
    for (const auto &elem : list) {
      os << sep << elem;
      sep = ',';
    }
  }
}

const Scope *ModuleDetails::parent() const {
  return isSubmodule_ && scope_ ? &scope_->parent() : nullptr;
}
const Scope *ModuleDetails::ancestor() const {
  return isSubmodule_ && scope_ ? FindModuleContaining(*scope_) : nullptr;
}
void ModuleDetails::set_scope(const Scope *scope) {
  CHECK(!scope_);
  bool scopeIsSubmodule{scope->parent().kind() == Scope::Kind::Module};
  CHECK(isSubmodule_ == scopeIsSubmodule);
  scope_ = scope;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const SubprogramDetails &x) {
  DumpBool(os, "isInterface", x.isInterface_);
  DumpExpr(os, "bindName", x.bindName_);
  if (x.result_) {
    DumpType(os << " result:", x.result());
    os << x.result_->name();
    if (!x.result_->attrs().empty()) {
      os << ", " << x.result_->attrs();
    }
  }
  if (x.entryScope_) {
    os << " entry";
    if (x.entryScope_->symbol()) {
      os << " in " << x.entryScope_->symbol()->name();
    }
  }
  char sep{'('};
  os << ' ';
  for (const Symbol *arg : x.dummyArgs_) {
    os << sep;
    sep = ',';
    if (arg) {
      DumpType(os, *arg);
      os << arg->name();
    } else {
      os << '*';
    }
  }
  os << (sep == '(' ? "()" : ")");
  return os;
}

void EntityDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = &type;
}

void EntityDetails::ReplaceType(const DeclTypeSpec &type) { type_ = &type; }

void ObjectEntityDetails::set_shape(const ArraySpec &shape) {
  CHECK(shape_.empty());
  for (const auto &shapeSpec : shape) {
    shape_.push_back(shapeSpec);
  }
}
void ObjectEntityDetails::set_coshape(const ArraySpec &coshape) {
  CHECK(coshape_.empty());
  for (const auto &shapeSpec : coshape) {
    coshape_.push_back(shapeSpec);
  }
}

ProcEntityDetails::ProcEntityDetails(EntityDetails &&d) : EntityDetails(d) {
  if (type()) {
    interface_.set_type(*type());
  }
}

const Symbol &UseDetails::module() const {
  // owner is a module so it must have a symbol:
  return *symbol_->owner().symbol();
}

UseErrorDetails::UseErrorDetails(const UseDetails &useDetails) {
  add_occurrence(useDetails.location(), *useDetails.module().scope());
}
UseErrorDetails &UseErrorDetails::add_occurrence(
    const SourceName &location, const Scope &module) {
  occurrences_.push_back(std::make_pair(location, &module));
  return *this;
}

GenericDetails::GenericDetails(const SymbolVector &specificProcs)
  : specificProcs_{specificProcs} {}

void GenericDetails::AddSpecificProc(
    const Symbol &proc, SourceName bindingName) {
  specificProcs_.push_back(proc);
  bindingNames_.push_back(bindingName);
}
void GenericDetails::set_specific(Symbol &specific) {
  CHECK(!specific_);
  CHECK(!derivedType_);
  specific_ = &specific;
}
void GenericDetails::set_derivedType(Symbol &derivedType) {
  CHECK(!specific_);
  CHECK(!derivedType_);
  derivedType_ = &derivedType;
}

const Symbol *GenericDetails::CheckSpecific() const {
  return const_cast<GenericDetails *>(this)->CheckSpecific();
}
Symbol *GenericDetails::CheckSpecific() {
  if (specific_) {
    for (const Symbol &proc : specificProcs_) {
      if (&proc == specific_) {
        return nullptr;
      }
    }
    return specific_;
  } else {
    return nullptr;
  }
}

void GenericDetails::CopyFrom(const GenericDetails &from) {
  if (from.specific_) {
    CHECK(!specific_ || specific_ == from.specific_);
    specific_ = from.specific_;
  }
  if (from.derivedType_) {
    CHECK(!derivedType_ || derivedType_ == from.derivedType_);
    derivedType_ = from.derivedType_;
  }
  for (const Symbol &symbol : from.specificProcs_) {
    if (std::find_if(specificProcs_.begin(), specificProcs_.end(),
            [&](const Symbol &mySymbol) { return &mySymbol == &symbol; }) ==
        specificProcs_.end()) {
      specificProcs_.push_back(symbol);
    }
  }
}

// The name of the kind of details for this symbol.
// This is primarily for debugging.
std::string DetailsToString(const Details &details) {
  return std::visit(
      common::visitors{
          [](const UnknownDetails &) { return "Unknown"; },
          [](const MainProgramDetails &) { return "MainProgram"; },
          [](const ModuleDetails &) { return "Module"; },
          [](const SubprogramDetails &) { return "Subprogram"; },
          [](const SubprogramNameDetails &) { return "SubprogramName"; },
          [](const EntityDetails &) { return "Entity"; },
          [](const ObjectEntityDetails &) { return "ObjectEntity"; },
          [](const ProcEntityDetails &) { return "ProcEntity"; },
          [](const DerivedTypeDetails &) { return "DerivedType"; },
          [](const UseDetails &) { return "Use"; },
          [](const UseErrorDetails &) { return "UseError"; },
          [](const HostAssocDetails &) { return "HostAssoc"; },
          [](const GenericDetails &) { return "Generic"; },
          [](const ProcBindingDetails &) { return "ProcBinding"; },
          [](const NamelistDetails &) { return "Namelist"; },
          [](const CommonBlockDetails &) { return "CommonBlockDetails"; },
          [](const FinalProcDetails &) { return "FinalProc"; },
          [](const TypeParamDetails &) { return "TypeParam"; },
          [](const MiscDetails &) { return "Misc"; },
          [](const AssocEntityDetails &) { return "AssocEntity"; },
      },
      details);
}

const std::string Symbol::GetDetailsName() const {
  return DetailsToString(details_);
}

void Symbol::set_details(Details &&details) {
  CHECK(CanReplaceDetails(details));
  details_ = std::move(details);
}

bool Symbol::CanReplaceDetails(const Details &details) const {
  if (has<UnknownDetails>()) {
    return true;  // can always replace UnknownDetails
  } else {
    return std::visit(
        common::visitors{
            [](const UseErrorDetails &) { return true; },
            [&](const ObjectEntityDetails &) { return has<EntityDetails>(); },
            [&](const ProcEntityDetails &) { return has<EntityDetails>(); },
            [&](const SubprogramDetails &) {
              return has<SubprogramNameDetails>() || has<EntityDetails>();
            },
            [&](const DerivedTypeDetails &) {
              auto *derived{detailsIf<DerivedTypeDetails>()};
              return derived && derived->isForwardReferenced();
            },
            [](const auto &) { return false; },
        },
        details);
  }
}

// Usually a symbol's name is the first occurrence in the source, but sometimes
// we want to replace it with one at a different location (but same characters).
void Symbol::ReplaceName(const SourceName &name) {
  CHECK(name == name_);
  name_ = name;
}

void Symbol::SetType(const DeclTypeSpec &type) {
  std::visit(
      common::visitors{
          [&](EntityDetails &x) { x.set_type(type); },
          [&](ObjectEntityDetails &x) { x.set_type(type); },
          [&](AssocEntityDetails &x) { x.set_type(type); },
          [&](ProcEntityDetails &x) { x.interface().set_type(type); },
          [&](TypeParamDetails &x) { x.set_type(type); },
          [](auto &) {},
      },
      details_);
}

bool Symbol::IsDummy() const {
  return std::visit(
      common::visitors{[](const EntityDetails &x) { return x.isDummy(); },
          [](const ObjectEntityDetails &x) { return x.isDummy(); },
          [](const ProcEntityDetails &x) { return x.isDummy(); },
          [](const HostAssocDetails &x) { return x.symbol().IsDummy(); },
          [](const auto &) { return false; }},
      details_);
}

bool Symbol::IsFuncResult() const {
  return std::visit(
      common::visitors{[](const EntityDetails &x) { return x.isFuncResult(); },
          [](const ObjectEntityDetails &x) { return x.isFuncResult(); },
          [](const ProcEntityDetails &x) { return x.isFuncResult(); },
          [](const HostAssocDetails &x) { return x.symbol().IsFuncResult(); },
          [](const auto &) { return false; }},
      details_);
}

bool Symbol::IsObjectArray() const {
  const auto *details{std::get_if<ObjectEntityDetails>(&details_)};
  return details && details->IsArray();
}

bool Symbol::IsSubprogram() const {
  return std::visit(
      common::visitors{
          [](const SubprogramDetails &) { return true; },
          [](const SubprogramNameDetails &) { return true; },
          [](const GenericDetails &) { return true; },
          [](const UseDetails &x) { return x.symbol().IsSubprogram(); },
          [](const auto &) { return false; },
      },
      details_);
}

bool Symbol::IsFromModFile() const {
  return test(Flag::ModFile) ||
      (!owner_->IsGlobal() && owner_->symbol()->IsFromModFile());
}

ObjectEntityDetails::ObjectEntityDetails(EntityDetails &&d)
  : EntityDetails(d) {}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const EntityDetails &x) {
  DumpBool(os, "dummy", x.isDummy());
  DumpBool(os, "funcResult", x.isFuncResult());
  if (x.type()) {
    os << " type: " << *x.type();
  }
  DumpExpr(os, "bindName", x.bindName_);
  return os;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const ObjectEntityDetails &x) {
  os << *static_cast<const EntityDetails *>(&x);
  DumpList(os, "shape", x.shape());
  DumpList(os, "coshape", x.coshape());
  DumpExpr(os, "init", x.init_);
  return os;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const AssocEntityDetails &x) {
  os << *static_cast<const EntityDetails *>(&x);
  DumpExpr(os, "expr", x.expr());
  return os;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const ProcEntityDetails &x) {
  if (auto *symbol{x.interface_.symbol()}) {
    os << ' ' << symbol->name();
  } else {
    DumpType(os, x.interface_.type());
  }
  DumpExpr(os, "bindName", x.bindName());
  DumpOptional(os, "passName", x.passName());
  if (x.init()) {
    if (const Symbol * target{*x.init()}) {
      os << " => " << target->name();
    } else {
      os << " => NULL()";
    }
  }
  return os;
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &os, const DerivedTypeDetails &x) {
  DumpBool(os, "sequence", x.sequence_);
  DumpList(os, "components", x.componentNames_);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Details &details) {
  os << DetailsToString(details);
  std::visit(
      common::visitors{
          [&](const UnknownDetails &) {},
          [&](const MainProgramDetails &) {},
          [&](const ModuleDetails &x) {
            if (x.isSubmodule()) {
              os << " (";
              if (x.ancestor()) {
                auto ancestor{x.ancestor()->GetName().value()};
                os << ancestor;
                if (x.parent()) {
                  auto parent{x.parent()->GetName().value()};
                  if (ancestor != parent) {
                    os << ':' << parent;
                  }
                }
              }
              os << ")";
            }
          },
          [&](const SubprogramNameDetails &x) {
            os << ' ' << EnumToString(x.kind());
          },
          [&](const UseDetails &x) {
            os << " from " << x.symbol().name() << " in " << x.module().name();
          },
          [&](const UseErrorDetails &x) {
            os << " uses:";
            for (const auto &[location, module] : x.occurrences()) {
              os << " from " << module->GetName().value() << " at " << location;
            }
          },
          [](const HostAssocDetails &) {},
          [&](const GenericDetails &x) {
            os << ' ' << x.kind().ToString();
            DumpBool(os, "(specific)", x.specific() != nullptr);
            DumpBool(os, "(derivedType)", x.derivedType() != nullptr);
            os << " procs:";
            DumpSymbolVector(os, x.specificProcs());
          },
          [&](const ProcBindingDetails &x) {
            os << " => " << x.symbol().name();
            DumpOptional(os, "passName", x.passName());
          },
          [&](const NamelistDetails &x) {
            os << ':';
            DumpSymbolVector(os, x.objects());
          },
          [&](const CommonBlockDetails &x) {
            os << ':';
            for (const Symbol &object : x.objects()) {
              os << ' ' << object.name();
            }
          },
          [&](const FinalProcDetails &) {},
          [&](const TypeParamDetails &x) {
            DumpOptional(os, "type", x.type());
            os << ' ' << common::EnumToString(x.attr());
            DumpExpr(os, "init", x.init());
          },
          [&](const MiscDetails &x) {
            os << ' ' << MiscDetails::EnumToString(x.kind());
          },
          [&](const auto &x) { os << x; },
      },
      details);
  return os;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &o, Symbol::Flag flag) {
  return o << Symbol::EnumToString(flag);
}

llvm::raw_ostream &operator<<(
    llvm::raw_ostream &o, const Symbol::Flags &flags) {
  std::size_t n{flags.count()};
  std::size_t seen{0};
  for (std::size_t j{0}; seen < n; ++j) {
    Symbol::Flag flag{static_cast<Symbol::Flag>(j)};
    if (flags.test(flag)) {
      if (seen++ > 0) {
        o << ", ";
      }
      o << flag;
    }
  }
  return o;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Symbol &symbol) {
  os << symbol.name();
  if (!symbol.attrs().empty()) {
    os << ", " << symbol.attrs();
  }
  if (!symbol.flags().empty()) {
    os << " (" << symbol.flags() << ')';
  }
  os << ": " << symbol.details_;
  return os;
}

// Output a unique name for a scope by qualifying it with the names of
// parent scopes. For scopes without corresponding symbols, use the kind
// with an index (e.g. Block1, Block2, etc.).
static void DumpUniqueName(llvm::raw_ostream &os, const Scope &scope) {
  if (!scope.IsGlobal()) {
    DumpUniqueName(os, scope.parent());
    os << '/';
    if (auto *scopeSymbol{scope.symbol()};
        scopeSymbol && !scopeSymbol->name().empty()) {
      os << scopeSymbol->name();
    } else {
      int index{1};
      for (auto &child : scope.parent().children()) {
        if (child == scope) {
          break;
        }
        if (child.kind() == scope.kind()) {
          ++index;
        }
      }
      os << Scope::EnumToString(scope.kind()) << index;
    }
  }
}

// Dump a symbol for UnparseWithSymbols. This will be used for tests so the
// format should be reasonably stable.
llvm::raw_ostream &DumpForUnparse(
    llvm::raw_ostream &os, const Symbol &symbol, bool isDef) {
  DumpUniqueName(os, symbol.owner());
  os << '/' << symbol.name();
  if (isDef) {
    if (!symbol.attrs().empty()) {
      os << ' ' << symbol.attrs();
    }
    if (!symbol.flags().empty()) {
      os << " (" << symbol.flags() << ')';
    }
    os << ' ' << symbol.GetDetailsName();
    DumpType(os, symbol.GetType());
  }
  return os;
}

const DerivedTypeSpec *Symbol::GetParentTypeSpec(const Scope *scope) const {
  if (const Symbol * parentComponent{GetParentComponent(scope)}) {
    const auto &object{parentComponent->get<ObjectEntityDetails>()};
    return &object.type()->derivedTypeSpec();
  } else {
    return nullptr;
  }
}

const Symbol *Symbol::GetParentComponent(const Scope *scope) const {
  if (const auto *dtDetails{detailsIf<DerivedTypeDetails>()}) {
    if (!scope) {
      scope = scope_;
    }
    return dtDetails->GetParentComponent(DEREF(scope));
  } else {
    return nullptr;
  }
}

// Utility routine for InstantiateComponent(): applies type
// parameter values to an intrinsic type spec.
static const DeclTypeSpec &InstantiateIntrinsicType(Scope &scope,
    const DeclTypeSpec &spec, SemanticsContext &semanticsContext) {
  const IntrinsicTypeSpec &intrinsic{DEREF(spec.AsIntrinsic())};
  if (evaluate::ToInt64(intrinsic.kind())) {
    return spec;  // KIND is already a known constant
  }
  // The expression was not originally constant, but now it must be so
  // in the context of a parameterized derived type instantiation.
  KindExpr copy{intrinsic.kind()};
  evaluate::FoldingContext &foldingContext{semanticsContext.foldingContext()};
  copy = evaluate::Fold(foldingContext, std::move(copy));
  int kind{semanticsContext.GetDefaultKind(intrinsic.category())};
  if (auto value{evaluate::ToInt64(copy)}) {
    if (evaluate::IsValidKindOfIntrinsicType(intrinsic.category(), *value)) {
      kind = *value;
    } else {
      foldingContext.messages().Say(
          "KIND parameter value (%jd) of intrinsic type %s "
          "did not resolve to a supported value"_err_en_US,
          static_cast<std::intmax_t>(*value),
          parser::ToUpperCaseLetters(
              common::EnumToString(intrinsic.category())));
    }
  }
  switch (spec.category()) {
  case DeclTypeSpec::Numeric:
    return scope.MakeNumericType(intrinsic.category(), KindExpr{kind});
  case DeclTypeSpec::Logical:  //
    return scope.MakeLogicalType(KindExpr{kind});
  case DeclTypeSpec::Character:
    return scope.MakeCharacterType(
        ParamValue{spec.characterTypeSpec().length()}, KindExpr{kind});
  default: CRASH_NO_CASE;
  }
}

Symbol &Symbol::InstantiateComponent(
    Scope &scope, SemanticsContext &context) const {
  auto &foldingContext{context.foldingContext()};
  auto pair{scope.try_emplace(name(), attrs())};
  Symbol &result{*pair.first->second};
  if (!pair.second) {
    // Symbol was already present in the scope, which can only happen
    // in the case of type parameters.
    CHECK(has<TypeParamDetails>());
    return result;
  }
  result.attrs() = attrs();
  result.flags() = flags();
  result.set_details(common::Clone(details()));
  if (auto *details{result.detailsIf<ObjectEntityDetails>()}) {
    if (DeclTypeSpec * origType{result.GetType()}) {
      if (const DerivedTypeSpec * derived{origType->AsDerived()}) {
        DerivedTypeSpec newSpec{*derived};
        newSpec.CookParameters(foldingContext);  // enables AddParamValue()
        if (test(Symbol::Flag::ParentComp)) {
          // Forward any explicit type parameter values from the
          // derived type spec under instantiation that define type parameters
          // of the parent component to the derived type spec of the
          // parent component.
          const DerivedTypeSpec &instanceSpec{
              DEREF(foldingContext.pdtInstance())};
          for (const auto &[name, value] : instanceSpec.parameters()) {
            if (scope.find(name) == scope.end()) {
              newSpec.AddParamValue(name, ParamValue{value});
            }
          }
        }
        details->ReplaceType(FindOrInstantiateDerivedType(
            scope, std::move(newSpec), context, origType->category()));
      } else if (origType->AsIntrinsic()) {
        details->ReplaceType(
            InstantiateIntrinsicType(scope, *origType, context));
      } else if (origType->category() != DeclTypeSpec::ClassStar) {
        DIE("instantiated component has type that is "
            "neither intrinsic, derived, nor CLASS(*)");
      }
    }
    details->set_init(
        evaluate::Fold(foldingContext, std::move(details->init())));
    for (ShapeSpec &dim : details->shape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(
            Fold(foldingContext, std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(
            Fold(foldingContext, std::move(dim.ubound().GetExplicit())));
      }
    }
    for (ShapeSpec &dim : details->coshape()) {
      if (dim.lbound().isExplicit()) {
        dim.lbound().SetExplicit(
            Fold(foldingContext, std::move(dim.lbound().GetExplicit())));
      }
      if (dim.ubound().isExplicit()) {
        dim.ubound().SetExplicit(
            Fold(foldingContext, std::move(dim.ubound().GetExplicit())));
      }
    }
  } else if (!attrs_.test(Attr::NOPASS)) {
    std::visit(
        [&result](const auto &x) {
          using Ty = std::decay_t<decltype(x)>;
          if constexpr (std::is_base_of_v<WithPassArg, Ty>) {
            if (auto passName{x.passName()}) {
              result.get<Ty>().set_passName(*passName);
            }
          }
        },
        details_);
  }
  return result;
}

void DerivedTypeDetails::add_component(const Symbol &symbol) {
  if (symbol.test(Symbol::Flag::ParentComp)) {
    CHECK(componentNames_.empty());
  }
  componentNames_.push_back(symbol.name());
}

const Symbol *DerivedTypeDetails::GetParentComponent(const Scope &scope) const {
  if (auto extends{GetParentComponentName()}) {
    if (auto iter{scope.find(*extends)}; iter != scope.cend()) {
      if (const Symbol & symbol{*iter->second};
          symbol.test(Symbol::Flag::ParentComp)) {
        return &symbol;
      }
    }
  }
  return nullptr;
}

void TypeParamDetails::set_type(const DeclTypeSpec &type) {
  CHECK(!type_);
  type_ = &type;
}

bool GenericKind::IsIntrinsicOperator() const {
  return Is(OtherKind::Concat) || Has<common::LogicalOperator>() ||
      Has<common::NumericOperator>() || Has<common::RelationalOperator>();
}

bool GenericKind::IsOperator() const {
  return IsDefinedOperator() || IsIntrinsicOperator();
}

std::string GenericKind::ToString() const {
  return std::visit(
      common::visitors {
        [](const OtherKind &x) { return EnumToString(x); },
            [](const DefinedIo &x) { return EnumToString(x); },
#if !__clang__ && __GNUC__ == 7 && __GNUC_MINOR__ == 2
            [](const common::NumericOperator &x) {
              return common::EnumToString(x);
            },
            [](const common::LogicalOperator &x) {
              return common::EnumToString(x);
            },
            [](const common::RelationalOperator &x) {
              return common::EnumToString(x);
            },
#else
            [](const auto &x) { return common::EnumToString(x); },
#endif
      },
      u);
}

bool GenericKind::Is(GenericKind::OtherKind x) const {
  const OtherKind *y{std::get_if<OtherKind>(&u)};
  return y && *y == x;
}

}
