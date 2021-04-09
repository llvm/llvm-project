//===-- Mangler.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Lower/Mangler.h"
#include "flang/Common/reference.h"
#include "flang/Lower/Todo.h"
#include "flang/Lower/Utils.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Semantics/tools.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

// recursively build the vector of module scopes
static void moduleNames(const Fortran::semantics::Scope &scope,
                        llvm::SmallVector<llvm::StringRef, 2> &result) {
  if (scope.kind() == Fortran::semantics::Scope::Kind::Global) {
    return;
  }
  moduleNames(scope.parent(), result);
  if (scope.kind() == Fortran::semantics::Scope::Kind::Module)
    if (auto *symbol = scope.symbol())
      result.emplace_back(toStringRef(symbol->name()));
}

static llvm::SmallVector<llvm::StringRef, 2>
moduleNames(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  llvm::SmallVector<llvm::StringRef, 2> result;
  moduleNames(scope, result);
  return result;
}

static llvm::Optional<llvm::StringRef>
hostName(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  if (scope.kind() == Fortran::semantics::Scope::Kind::Subprogram) {
    assert(scope.symbol() && "subprogram scope must have a symbol");
    return {toStringRef(scope.symbol()->name())};
  }
  return {};
}

static const Fortran::semantics::Symbol *
findInterfaceIfSeperateMP(const Fortran::semantics::Symbol &symbol) {
  const auto &scope = symbol.owner();
  if (symbol.attrs().test(Fortran::semantics::Attr::MODULE) &&
      scope.IsSubmodule()) {
    // FIXME symbol from MpSubprogramStmt do not seem to have
    // Attr::MODULE set.
    const auto *iface = scope.parent().FindSymbol(symbol.name());
    assert(iface && "Separate module procedure must be declared");
    return iface;
  }
  return nullptr;
}

// Mangle the name of `symbol` to make it unique within FIR's symbol table using
// the FIR name mangler, `mangler`
std::string
Fortran::lower::mangle::mangleName(const Fortran::semantics::Symbol &symbol,
                                   bool keepExternalInScope) {
  // Resolve host and module association before mangling
  const auto &ultimateSymbol = symbol.GetUltimate();
  auto symbolName = toStringRef(ultimateSymbol.name());

  return std::visit(
      Fortran::common::visitors{
          [&](const Fortran::semantics::MainProgramDetails &) {
            return fir::NameUniquer::doProgramEntry().str();
          },
          [&](const Fortran::semantics::SubprogramDetails &) {
            // Mangle external procedure without any scope prefix.
            if (!keepExternalInScope &&
                Fortran::semantics::IsExternal(ultimateSymbol))
              return fir::NameUniquer::doProcedure(llvm::None, llvm::None,
                                                   symbolName);
            // Separate module subprograms must be mangled according to the
            // scope where they were declared (the symbol we have is the
            // definition).
            const auto *interface = &ultimateSymbol;
            if (const auto *mpIface = findInterfaceIfSeperateMP(ultimateSymbol))
              interface = mpIface;
            auto modNames = moduleNames(*interface);
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
            return fir::NameUniquer::doProcedure(llvm::None, llvm::None,
                                                 symbolName);
          },
          [&](const Fortran::semantics::ObjectEntityDetails &) {
            auto modNames = moduleNames(ultimateSymbol);
            auto optHost = hostName(ultimateSymbol);
            if (Fortran::semantics::IsNamedConstant(ultimateSymbol))
              return fir::NameUniquer::doConstant(modNames, optHost,
                                                  symbolName);
            return fir::NameUniquer::doVariable(modNames, optHost, symbolName);
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
          [](const auto &) -> std::string { TODO_NOLOC("symbol mangling"); },
      },
      ultimateSymbol.details());
}

std::string Fortran::lower::mangle::mangleName(
    const Fortran::semantics::DerivedTypeSpec &derivedType) {
  // Resolve host and module association before mangling
  const auto &ultimateSymbol = derivedType.typeSymbol().GetUltimate();
  auto symbolName = toStringRef(ultimateSymbol.name());
  auto modNames = moduleNames(ultimateSymbol);
  auto optHost = hostName(ultimateSymbol);
  llvm::SmallVector<std::int64_t> kinds;
  for (const auto &param :
       Fortran::semantics::OrderParameterDeclarations(ultimateSymbol)) {
    const auto &paramDetails =
        param->get<Fortran::semantics::TypeParamDetails>();
    if (paramDetails.attr() == Fortran::common::TypeParamAttr::Kind) {
      const auto *paramValue = derivedType.FindParameter(param->name());
      assert(paramValue && "derived type kind parameter value not found");
      auto paramExpr = paramValue->GetExplicit();
      assert(paramExpr && "derived type kind param not explicit");
      auto init = Fortran::evaluate::ToInt64(paramValue->GetExplicit());
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
// Intrinsic Procedure Mangling
//===----------------------------------------------------------------------===//

/// Helper to encode type into string for intrinsic procedure names.
/// Note: mlir has Type::dump(ostream) methods but it may add "!" that is not
/// suitable for function names.
static std::string typeToString(mlir::Type t) {
  if (auto refT{t.dyn_cast<fir::ReferenceType>()})
    return "ref_" + typeToString(refT.getEleTy());
  if (auto i{t.dyn_cast<mlir::IntegerType>()}) {
    return "i" + std::to_string(i.getWidth());
  }
  if (auto cplx{t.dyn_cast<fir::ComplexType>()}) {
    return "z" + std::to_string(cplx.getFKind());
  }
  if (auto real{t.dyn_cast<fir::RealType>()}) {
    return "r" + std::to_string(real.getFKind());
  }
  if (auto f{t.dyn_cast<mlir::FloatType>()}) {
    return "f" + std::to_string(f.getWidth());
  }
  if (auto logical{t.dyn_cast<fir::LogicalType>()}) {
    return "l" + std::to_string(logical.getFKind());
  }
  if (auto character{t.dyn_cast<fir::CharacterType>()}) {
    return "c" + std::to_string(character.getFKind());
  }
  if (auto boxCharacter{t.dyn_cast<fir::BoxCharType>()}) {
    return "bc" + std::to_string(boxCharacter.getEleTy().getFKind());
  }
  llvm_unreachable("no mangling for type");
}

std::string fir::mangleIntrinsicProcedure(llvm::StringRef intrinsic,
                                          mlir::FunctionType funTy) {
  std::string name = "fir.";
  name.append(intrinsic.str()).append(".");
  assert(funTy.getNumResults() == 1 && "only function mangling supported");
  name.append(typeToString(funTy.getResult(0)));
  auto e = funTy.getNumInputs();
  for (decltype(e) i = 0; i < e; ++i)
    name.append(".").append(typeToString(funTy.getInput(i)));
  return name;
}
