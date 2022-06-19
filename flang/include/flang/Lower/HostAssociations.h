//===-- Lower/HostAssociations.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_HOSTASSOCIATIONS_H
#define FORTRAN_LOWER_HOSTASSOCIATIONS_H

#include "mlir/IR/Location.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SetVector.h"

namespace Fortran {
namespace semantics {
class Symbol;
}

namespace lower {
class AbstractConverter;
class SymMap;

/// Internal procedures in Fortran may access variables declared in the host
/// procedure directly. We bundle these variables together in a tuple and pass
/// them as an extra argument.
class HostAssociations {
public:
  /// Returns true iff there are no host associations.
  bool empty() const { return symbols.empty(); }

  /// Adds a set of Symbols that will be the host associated bindings for this
  /// host procedure.
  void addSymbolsToBind(
      const llvm::SetVector<const Fortran::semantics::Symbol *> &s) {
    assert(empty() && "symbol set must be initially empty");
    symbols = s;
  }

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
    return symbols.contains(&symbol);
  }

private:
  /// Canonical vector of host associated symbols.
  llvm::SetVector<const Fortran::semantics::Symbol *> symbols;

  /// The type of the extra argument to be added to each internal procedure.
  mlir::Type argType;
};
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_HOSTASSOCIATIONS_H
