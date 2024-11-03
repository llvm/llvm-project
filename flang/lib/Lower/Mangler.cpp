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
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MD5.h"
#include <optional>

// recursively build the vector of module scopes
static void moduleNames(const Fortran::semantics::Scope &scope,
                        llvm::SmallVector<llvm::StringRef> &result) {
  if (scope.IsTopLevel())
    return;
  moduleNames(scope.parent(), result);
  if (scope.kind() == Fortran::semantics::Scope::Kind::Module)
    if (const Fortran::semantics::Symbol *symbol = scope.symbol())
      result.emplace_back(toStringRef(symbol->name()));
}

static llvm::SmallVector<llvm::StringRef>
moduleNames(const Fortran::semantics::Symbol &symbol) {
  const Fortran::semantics::Scope &scope = symbol.owner();
  llvm::SmallVector<llvm::StringRef> result;
  moduleNames(scope, result);
  return result;
}

static std::optional<llvm::StringRef>
hostName(const Fortran::semantics::Symbol &symbol) {
  const Fortran::semantics::Scope &scope = symbol.owner();
  if (scope.kind() == Fortran::semantics::Scope::Kind::Subprogram) {
    assert(scope.symbol() && "subprogram scope must have a symbol");
    return toStringRef(scope.symbol()->name());
  }
  if (scope.kind() == Fortran::semantics::Scope::Kind::MainProgram)
    // Do not use the main program name, if any, because it may lead to name
    // collision with procedures with the same name in other compilation units
    // (technically illegal, but all compilers are able to compile and link
    // properly these programs).
    return llvm::StringRef("");
  return {};
}

// Mangle the name of `symbol` to make it unique within FIR's symbol table using
// the FIR name mangler, `mangler`
std::string
Fortran::lower::mangle::mangleName(const Fortran::semantics::Symbol &symbol,
                                   bool keepExternalInScope) {
  // Resolve host and module association before mangling
  const auto &ultimateSymbol = symbol.GetUltimate();
  auto symbolName = toStringRef(ultimateSymbol.name());

  // The Fortran and BIND(C) namespaces are counterintuitive. A
  // BIND(C) name is substituted early having precedence over the
  // Fortran name of the subprogram. By side-effect, this allows
  // multiple subprocedures with identical Fortran names to be legally
  // present in the program. Assume the BIND(C) name is unique.
  if (auto *overrideName = ultimateSymbol.GetBindName())
    return *overrideName;
  // TODO: the case of procedure that inherits the BIND(C) through another
  // interface (procedure(iface)), should be dealt within GetBindName()
  // directly, or some semantics wrapper.
  if (!Fortran::semantics::IsPointer(ultimateSymbol) &&
      Fortran::semantics::IsBindCProcedure(ultimateSymbol) &&
      Fortran::semantics::ClassifyProcedure(symbol) !=
          Fortran::semantics::ProcedureDefinitionClass::Internal)
    return ultimateSymbol.name().ToString();

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
            assert(interface && "Separate module procedure must be declared");
            llvm::SmallVector<llvm::StringRef> modNames =
                moduleNames(*interface);
            return fir::NameUniquer::doProcedure(modNames, hostName(*interface),
                                                 symbolName);
          },
          [&](const Fortran::semantics::ProcEntityDetails &) {
            // Mangle procedure pointers and dummy procedures as variables
            if (Fortran::semantics::IsPointer(ultimateSymbol) ||
                Fortran::semantics::IsDummy(ultimateSymbol))
              return fir::NameUniquer::doVariable(moduleNames(ultimateSymbol),
                                                  hostName(ultimateSymbol),
                                                  symbolName);
            // Otherwise, this is an external procedure, even if it does not
            // have an explicit EXTERNAL attribute. Mangle it without any
            // prefix.
            return fir::NameUniquer::doProcedure(std::nullopt, std::nullopt,
                                                 symbolName);
          },
          [&](const Fortran::semantics::ObjectEntityDetails &) {
            llvm::SmallVector<llvm::StringRef> modNames =
                moduleNames(ultimateSymbol);
            std::optional<llvm::StringRef> optHost = hostName(ultimateSymbol);
            if (Fortran::semantics::IsNamedConstant(ultimateSymbol))
              return fir::NameUniquer::doConstant(modNames, optHost,
                                                  symbolName);
            return fir::NameUniquer::doVariable(modNames, optHost, symbolName);
          },
          [&](const Fortran::semantics::NamelistDetails &) {
            llvm::SmallVector<llvm::StringRef> modNames =
                moduleNames(ultimateSymbol);
            std::optional<llvm::StringRef> optHost = hostName(ultimateSymbol);
            return fir::NameUniquer::doNamelistGroup(modNames, optHost,
                                                     symbolName);
          },
          [&](const Fortran::semantics::CommonBlockDetails &) {
            return fir::NameUniquer::doCommonBlock(symbolName);
          },
          [&](const Fortran::semantics::DerivedTypeDetails &) -> std::string {
            // Derived type mangling must used mangleName(DerivedTypeSpec&) so
            // that kind type parameter values can be mangled.
            llvm::report_fatal_error(
                "only derived type instances can be mangled");
          },
          [&](const Fortran::semantics::ProcBindingDetails &procBinding)
              -> std::string {
            return mangleName(procBinding.symbol(), keepExternalInScope);
          },
          [](const auto &) -> std::string { TODO_NOLOC("symbol mangling"); },
      },
      ultimateSymbol.details());
}

std::string Fortran::lower::mangle::mangleName(
    const Fortran::semantics::DerivedTypeSpec &derivedType) {
  // Resolve host and module association before mangling
  const Fortran::semantics::Symbol &ultimateSymbol =
      derivedType.typeSymbol().GetUltimate();
  llvm::StringRef symbolName = toStringRef(ultimateSymbol.name());
  llvm::SmallVector<llvm::StringRef> modNames = moduleNames(ultimateSymbol);
  std::optional<llvm::StringRef> optHost = hostName(ultimateSymbol);
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
  return fir::NameUniquer::doType(modNames, optHost, symbolName, kinds);
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
    const uint8_t *addr, size_t size,
    const Fortran::evaluate::ConstantSubscripts &shape,
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
    return name += "null";
  llvm::MD5 hashValue{};
  hashValue.update(llvm::ArrayRef<uint8_t>{addr, size});
  llvm::MD5::MD5Result hashResult;
  hashValue.final(hashResult);
  llvm::SmallString<32> hashString;
  llvm::MD5::stringifyResult(hashResult, hashString);
  return name += hashString.c_str();
}

std::string Fortran::lower::mangle::globalNamelistDescriptorName(
    const Fortran::semantics::Symbol &sym) {
  std::string name = mangleName(sym);
  return IsAllocatableOrPointer(sym) ? name : name + ".desc"s;
}
