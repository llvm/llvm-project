//===-- Lower/Mangler.h -- name mangling ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_MANGLER_H
#define FORTRAN_LOWER_MANGLER_H

#include "flang/Evaluate/expression.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace Fortran {
namespace common {
template <typename>
class Reference;
}

namespace semantics {
class Scope;
class Symbol;
class DerivedTypeSpec;
} // namespace semantics

namespace lower::mangle {

using ScopeBlockIdMap =
    llvm::DenseMap<Fortran::semantics::Scope *, std::int64_t>;

/// Convert a front-end symbol to a unique internal name.
/// A symbol that could be in a block scope must provide a ScopeBlockIdMap.
/// If \p keepExternalInScope is true, mangling an external symbol retains
/// the scope of the symbol. This is useful when setting the attributes of
/// a symbol where all the Fortran context is needed. Otherwise, external
/// symbols are mangled outside of any scope.
std::string mangleName(const semantics::Symbol &, ScopeBlockIdMap &,
                       bool keepExternalInScope = false,
                       bool underscoring = true);
std::string mangleName(const semantics::Symbol &,
                       bool keepExternalInScope = false,
                       bool underscoring = true);

/// Convert a derived type instance to an internal name.
std::string mangleName(const semantics::DerivedTypeSpec &, ScopeBlockIdMap &);

/// Add a scope specific mangling prefix to a compiler generated name.
std::string mangleName(std::string &, const Fortran::semantics::Scope &,
                       ScopeBlockIdMap &);

/// Recover the bare name of the original symbol from an internal name.
std::string demangleName(llvm::StringRef name);

std::string
mangleArrayLiteral(size_t size,
                   const Fortran::evaluate::ConstantSubscripts &shape,
                   Fortran::common::TypeCategory cat, int kind = 0,
                   Fortran::common::ConstantSubscript charLen = -1,
                   llvm::StringRef derivedName = {});

template <Fortran::common::TypeCategory TC, int KIND>
std::string mangleArrayLiteral(
    mlir::Type,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>> &x) {
  return mangleArrayLiteral(x.values().size() * sizeof(x.values()[0]),
                            x.shape(), TC, KIND);
}

template <int KIND>
std::string
mangleArrayLiteral(mlir::Type,
                   const Fortran::evaluate::Constant<Fortran::evaluate::Type<
                       Fortran::common::TypeCategory::Character, KIND>> &x) {
  return mangleArrayLiteral(x.values().size() * sizeof(x.values()[0]),
                            x.shape(), Fortran::common::TypeCategory::Character,
                            KIND, x.LEN());
}

inline std::string mangleArrayLiteral(
    mlir::Type eleTy,
    const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &x) {
  return mangleArrayLiteral(x.values().size() * sizeof(x.values()[0]),
                            x.shape(), Fortran::common::TypeCategory::Derived,
                            /*kind=*/0, /*charLen=*/-1,
                            mlir::cast<fir::RecordType>(eleTy).getName());
}

/// Return the compiler-generated name of a static namelist variable descriptor.
std::string globalNamelistDescriptorName(const Fortran::semantics::Symbol &sym);

/// Return the field name for a derived type component inside a fir.record type.
/// It is the component name if the component is not private. Otherwise it is
/// mangled with the component parent type to avoid any name clashes in type
/// extensions.
std::string getRecordTypeFieldName(const Fortran::semantics::Symbol &component,
                                   ScopeBlockIdMap &);

} // namespace lower::mangle
} // namespace Fortran

#endif // FORTRAN_LOWER_MANGLER_H
