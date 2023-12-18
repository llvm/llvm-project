//===-- Mangler.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Mangler.h"
#include "flang/Common/reference.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MD5.h"

/// Return all ancestor module and submodule scope names; all host procedure
/// and statement function scope names; and the innermost blockId containing
/// \p scope, including scope itself.
static std::tuple<llvm::SmallVector<llvm::StringRef>,
                  llvm::SmallVector<llvm::StringRef>, std::int64_t>
ancestors(const Fortran::semantics::Scope &scope,
          Fortran::lower::mangle::ScopeBlockIdMap &scopeBlockIdMap) {
  llvm::SmallVector<const Fortran::semantics::Scope *> scopes;
  for (auto *scp = &scope; !scp->IsGlobal(); scp = &scp->parent())
    scopes.push_back(scp);
  llvm::SmallVector<llvm::StringRef> modules;
  llvm::SmallVector<llvm::StringRef> procs;
  std::int64_t blockId = 0;
  for (auto iter = scopes.rbegin(), rend = scopes.rend(); iter != rend;
       ++iter) {
    auto *scp = *iter;
    switch (scp->kind()) {
    case Fortran::semantics::Scope::Kind::Module:
      modules.emplace_back(toStringRef(scp->symbol()->name()));
      break;
    case Fortran::semantics::Scope::Kind::Subprogram:
      procs.emplace_back(toStringRef(scp->symbol()->name()));
      break;
    case Fortran::semantics::Scope::Kind::MainProgram:
      // Do not use the main program name, if any, because it may collide
      // with a procedure of the same name in another compilation unit.
      // This is nonconformant, but universally allowed.
      procs.emplace_back(llvm::StringRef(""));
      break;
    case Fortran::semantics::Scope::Kind::BlockConstruct: {
      auto it = scopeBlockIdMap.find(scp);
      assert(it != scopeBlockIdMap.end() && it->second &&
             "invalid block identifier");
      blockId = it->second;
    } break;
    default:
      break;
    }
  }
  return {modules, procs, blockId};
}

/// Return all ancestor module and submodule scope names; all host procedure
/// and statement function scope names; and the innermost blockId containing
/// \p symbol.
static std::tuple<llvm::SmallVector<llvm::StringRef>,
                  llvm::SmallVector<llvm::StringRef>, std::int64_t>
ancestors(const Fortran::semantics::Symbol &symbol,
          Fortran::lower::mangle::ScopeBlockIdMap &scopeBlockIdMap) {
  return ancestors(symbol.owner(), scopeBlockIdMap);
}

/// Return a globally unique string for a compiler generated \p name.
std::string
Fortran::lower::mangle::mangleName(std::string &name,
                                   const Fortran::semantics::Scope &scope,
                                   ScopeBlockIdMap &scopeBlockIdMap) {
  llvm::SmallVector<llvm::StringRef> modules;
  llvm::SmallVector<llvm::StringRef> procs;
  std::int64_t blockId;
  std::tie(modules, procs, blockId) = ancestors(scope, scopeBlockIdMap);
  return fir::NameUniquer::doGenerated(modules, procs, blockId, name);
}

// Mangle the name of \p symbol to make it globally unique.
std::string Fortran::lower::mangle::mangleName(
    const Fortran::semantics::Symbol &symbol, ScopeBlockIdMap &scopeBlockIdMap,
    bool keepExternalInScope, bool underscoring) {
  // Resolve module and host associations before mangling.
  const auto &ultimateSymbol = symbol.GetUltimate();

  // The Fortran and BIND(C) namespaces are counterintuitive. A BIND(C) name is
  // substituted early, and has precedence over the Fortran name. This allows
  // multiple procedures or objects with identical Fortran names to legally
  // coexist. The BIND(C) name is unique.
  if (auto *overrideName = ultimateSymbol.GetBindName())
    return *overrideName;

  llvm::StringRef symbolName = toStringRef(ultimateSymbol.name());
  llvm::SmallVector<llvm::StringRef> modules;
  llvm::SmallVector<llvm::StringRef> procs;
  std::int64_t blockId;

  // mangle ObjectEntityDetails or AssocEntityDetails symbols.
  auto mangleObject = [&]() -> std::string {
    std::tie(modules, procs, blockId) =
        ancestors(ultimateSymbol, scopeBlockIdMap);
    if (Fortran::semantics::IsNamedConstant(ultimateSymbol))
      return fir::NameUniquer::doConstant(modules, procs, blockId, symbolName);
    return fir::NameUniquer::doVariable(modules, procs, blockId, symbolName);
  };

  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::semantics::MainProgramDetails &) {
            return fir::NameUniquer::doProgramEntry().str();
          },
          [&](const Fortran::semantics::SubprogramDetails &subpDetails) {
            // Mangle external procedure without any scope prefix.
            if (!keepExternalInScope &&
                Fortran::semantics::IsExternal(ultimateSymbol))
              return fir::NameUniquer::doProcedure(std::nullopt, std::nullopt,
                                                   symbolName);
            // A separate module procedure must be mangled according to its
            // declaration scope, not its definition scope.
            const Fortran::semantics::Symbol *interface = &ultimateSymbol;
            if (interface->attrs().test(Fortran::semantics::Attr::MODULE) &&
                interface->owner().IsSubmodule() && !subpDetails.isInterface())
              interface = subpDetails.moduleInterface();
            std::tie(modules, procs, blockId) = ancestors(
                interface ? *interface : ultimateSymbol, scopeBlockIdMap);
            return fir::NameUniquer::doProcedure(modules, procs, symbolName);
          },
          [&](const Fortran::semantics::ProcEntityDetails &) {
            // Mangle procedure pointers and dummy procedures as variables.
            if (Fortran::semantics::IsPointer(ultimateSymbol) ||
                Fortran::semantics::IsDummy(ultimateSymbol)) {
              std::tie(modules, procs, blockId) =
                  ancestors(ultimateSymbol, scopeBlockIdMap);
              return fir::NameUniquer::doVariable(modules, procs, blockId,
                                                  symbolName);
            }
            // Otherwise, this is an external procedure, with or without an
            // explicit EXTERNAL attribute. Mangle it without any prefix.
            return fir::NameUniquer::doProcedure(std::nullopt, std::nullopt,
                                                 symbolName);
          },
          [&](const Fortran::semantics::ObjectEntityDetails &) {
            return mangleObject();
          },
          [&](const Fortran::semantics::AssocEntityDetails &) {
            return mangleObject();
          },
          [&](const Fortran::semantics::NamelistDetails &) {
            std::tie(modules, procs, blockId) =
                ancestors(ultimateSymbol, scopeBlockIdMap);
            return fir::NameUniquer::doNamelistGroup(modules, procs,
                                                     symbolName);
          },
          [&](const Fortran::semantics::CommonBlockDetails &) {
            return Fortran::semantics::GetCommonBlockObjectName(ultimateSymbol,
                                                                underscoring);
          },
          [&](const Fortran::semantics::ProcBindingDetails &procBinding) {
            return mangleName(procBinding.symbol(), scopeBlockIdMap,
                              keepExternalInScope, underscoring);
          },
          [&](const Fortran::semantics::DerivedTypeDetails &) -> std::string {
            // Derived type mangling must use mangleName(DerivedTypeSpec) so
            // that kind type parameter values can be mangled.
            llvm::report_fatal_error(
                "only derived type instances can be mangled");
          },
          [](const auto &) -> std::string { TODO_NOLOC("symbol mangling"); },
      },
      ultimateSymbol.details());
}

std::string
Fortran::lower::mangle::mangleName(const Fortran::semantics::Symbol &symbol,
                                   bool keepExternalInScope,
                                   bool underscoring) {
  assert((symbol.owner().kind() !=
              Fortran::semantics::Scope::Kind::BlockConstruct ||
          symbol.has<Fortran::semantics::SubprogramDetails>()) &&
         "block object mangling must specify a scopeBlockIdMap");
  ScopeBlockIdMap scopeBlockIdMap;
  return mangleName(symbol, scopeBlockIdMap, keepExternalInScope, underscoring);
}

std::string Fortran::lower::mangle::mangleName(
    const Fortran::semantics::DerivedTypeSpec &derivedType,
    ScopeBlockIdMap &scopeBlockIdMap) {
  // Resolve module and host associations before mangling.
  const Fortran::semantics::Symbol &ultimateSymbol =
      derivedType.typeSymbol().GetUltimate();

  llvm::StringRef symbolName = toStringRef(ultimateSymbol.name());
  llvm::SmallVector<llvm::StringRef> modules;
  llvm::SmallVector<llvm::StringRef> procs;
  std::int64_t blockId;
  std::tie(modules, procs, blockId) =
      ancestors(ultimateSymbol, scopeBlockIdMap);
  llvm::SmallVector<std::int64_t> kinds;
  for (const auto &param :
       Fortran::semantics::OrderParameterDeclarations(ultimateSymbol)) {
    const auto &paramDetails =
        param->get<Fortran::semantics::TypeParamDetails>();
    if (paramDetails.attr() == Fortran::common::TypeParamAttr::Kind) {
      const Fortran::semantics::ParamValue *paramValue =
          derivedType.FindParameter(param->name());
      assert(paramValue && "derived type kind parameter value not found");
      const Fortran::semantics::MaybeIntExpr paramExpr =
          paramValue->GetExplicit();
      assert(paramExpr && "derived type kind param not explicit");
      std::optional<int64_t> init =
          Fortran::evaluate::ToInt64(paramValue->GetExplicit());
      assert(init && "derived type kind param is not constant");
      kinds.emplace_back(*init);
    }
  }
  return fir::NameUniquer::doType(modules, procs, blockId, symbolName, kinds);
}

std::string Fortran::lower::mangle::getRecordTypeFieldName(
    const Fortran::semantics::Symbol &component,
    ScopeBlockIdMap &scopeBlockIdMap) {
  if (!component.attrs().test(Fortran::semantics::Attr::PRIVATE))
    return component.name().ToString();
  const Fortran::semantics::DerivedTypeSpec *componentParentType =
      component.owner().derivedTypeSpec();
  assert(componentParentType &&
         "failed to retrieve private component parent type");
  // Do not mangle Iso C C_PTR and C_FUNPTR components. This type cannot be
  // extended as per Fortran 2018 7.5.7.1, mangling them makes the IR unreadable
  // when using ISO C modules, and lowering needs to know the component way
  // without access to semantics::Symbol.
  if (Fortran::semantics::IsIsoCType(componentParentType))
    return component.name().ToString();
  return mangleName(*componentParentType, scopeBlockIdMap) + "." +
         component.name().ToString();
}

std::string Fortran::lower::mangle::demangleName(llvm::StringRef name) {
  auto result = fir::NameUniquer::deconstruct(name);
  return result.second.name;
}

//===----------------------------------------------------------------------===//
// Array Literals Mangling
//===----------------------------------------------------------------------===//

static std::string typeToString(Fortran::common::TypeCategory cat, int kind,
                                llvm::StringRef derivedName) {
  switch (cat) {
  case Fortran::common::TypeCategory::Integer:
    return "i" + std::to_string(kind);
  case Fortran::common::TypeCategory::Real:
    return "r" + std::to_string(kind);
  case Fortran::common::TypeCategory::Complex:
    return "z" + std::to_string(kind);
  case Fortran::common::TypeCategory::Logical:
    return "l" + std::to_string(kind);
  case Fortran::common::TypeCategory::Character:
    return "c" + std::to_string(kind);
  case Fortran::common::TypeCategory::Derived:
    return derivedName.str();
  }
  llvm_unreachable("bad TypeCategory");
}

std::string Fortran::lower::mangle::mangleArrayLiteral(
    size_t size, const Fortran::evaluate::ConstantSubscripts &shape,
    Fortran::common::TypeCategory cat, int kind,
    Fortran::common::ConstantSubscript charLen, llvm::StringRef derivedName) {
  std::string typeId;
  for (Fortran::evaluate::ConstantSubscript extent : shape)
    typeId.append(std::to_string(extent)).append("x");
  if (charLen >= 0)
    typeId.append(std::to_string(charLen)).append("x");
  typeId.append(typeToString(cat, kind, derivedName));
  std::string name =
      fir::NameUniquer::doGenerated("ro."s.append(typeId).append("."));
  if (!size)
    name += "null.";
  return name;
}

std::string Fortran::lower::mangle::globalNamelistDescriptorName(
    const Fortran::semantics::Symbol &sym) {
  std::string name = mangleName(sym);
  return IsAllocatableOrObjectPointer(&sym) ? name : name + ".desc"s;
}
