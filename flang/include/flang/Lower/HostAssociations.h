//===-- Lower/HostAssociations.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_HOSTASSOCIATIONS_H
#define FORTRAN_LOWER_HOSTASSOCIATIONS_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"

namespace Fortran {
namespace semantics {
class Symbol;
} // namespace semantics

namespace lower {
class AbstractConverter;
class SymMap;
namespace pft {
struct FunctionLikeUnit;
struct Program;
struct Variable;
} // namespace pft

/// Internal procedures in Fortran may access variables declared in the host
/// procedure directly. We bundle these variables together in a tuple and pass
/// them as an extra argument.
class HostAssociations {
public:
  /// Returns true iff there are no host associations.
  bool empty() const { return tupleSymbols.empty() && globalSymbols.empty(); }

  /// Returns true iff there are host associations that are conveyed through
  /// an extra tuple argument.
  bool hasTupleAssociations() const { return !tupleSymbols.empty(); }

  /// True if internal procedures access captured variables in global storage.
  bool hasGlobalAssociations() const { return !globalSymbols.empty(); }

  /// Adds a set of Symbols that will be the host associated bindings for this
  /// host procedure.
  /// Storage classification uses program's selected MAIN globals.
  void addSymbolsToBind(
      const llvm::SetVector<const Fortran::semantics::Symbol *> &symbols,
      const pft::Program &program);

  /// Cache the variable list for a host with captured globals and select the
  /// bindings needed by its internal procedures.
  /// Call before lowering the host's first entry point.
  void prepareGlobalBindings(const pft::FunctionLikeUnit &host);

  /// Clear binding pointers before discarding the host's cached variable list.
  /// Call after lowering the host and all its internal procedures.
  void releaseGlobalBindings(const pft::FunctionLikeUnit &host);

  /// Code gen the FIR for the local bindings for the host associated symbols
  /// for the host (parent) procedure using `builder`.
  void hostProcedureBindings(AbstractConverter &converter, SymMap &symMap);

  /// Code gen the FIR for the local bindings for the host associated symbols
  /// for an internal (child) procedure using `builder`.
  void internalProcedureBindings(AbstractConverter &converter, SymMap &symMap);

  /// Return the type of the extra argument to add to each internal procedure.
  mlir::Type getArgumentType(AbstractConverter &convert);

  /// Is \p symbol host associated ?
  bool isAssociated(const Fortran::semantics::Symbol &symbol) const {
    return tupleSymbols.contains(&symbol) || globalSymbols.contains(&symbol);
  }

  /// Is the canonical symbol captured as a global instead of a tuple entry?
  bool isAssociatedGlobal(const Fortran::semantics::Symbol &symbol) const {
    return globalSymbols.contains(&symbol);
  }

private:
  /// Canonical vector of host associated local symbols.
  llvm::SetVector<const Fortran::semantics::Symbol *> tupleSymbols;

  /// Canonical vector of host associated global symbols.
  llvm::SetVector<const Fortran::semantics::Symbol *> globalSymbols;

  /// Captured globals and EQUIVALENCE storage, in reverse list order.
  /// Pointers refer to the host's cached list until releaseGlobalBindings().
  llvm::SmallVector<const pft::Variable *, 0> globalBindings;

  /// The type of the extra argument to be added to each internal procedure.
  mlir::Type argType;
};
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_HOSTASSOCIATIONS_H
