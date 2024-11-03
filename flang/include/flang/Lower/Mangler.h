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
class Symbol;
class DerivedTypeSpec;
} // namespace semantics

namespace lower::mangle {

/// Convert a front-end Symbol to an internal name.
/// If \p keepExternalInScope is true, the mangling of external symbols
/// retains the scope of the symbol declaring externals. Otherwise,
/// external symbols are mangled outside of any scope. Keeping the scope is
/// useful in attributes where all the Fortran context is to be maintained.
std::string mangleName(const semantics::Symbol &,
                       bool keepExternalInScope = false);

/// Convert a derived type instance to an internal name.
std::string mangleName(const semantics::DerivedTypeSpec &);

/// Recover the bare name of the original symbol from an internal name.
std::string demangleName(llvm::StringRef name);

std::string
mangleArrayLiteral(const uint8_t *addr, size_t size,
                   const Fortran::evaluate::ConstantSubscripts &shape,
                   Fortran::common::TypeCategory cat, int kind = 0,
                   Fortran::common::ConstantSubscript charLen = -1,
                   llvm::StringRef derivedName = {});

template <Fortran::common::TypeCategory TC, int KIND>
std::string mangleArrayLiteral(
    mlir::Type,
    const Fortran::evaluate::Constant<Fortran::evaluate::Type<TC, KIND>> &x) {
  return mangleArrayLiteral(
      reinterpret_cast<const uint8_t *>(x.values().data()),
      x.values().size() * sizeof(x.values()[0]), x.shape(), TC, KIND);
}

template <int KIND>
std::string
mangleArrayLiteral(mlir::Type,
                   const Fortran::evaluate::Constant<Fortran::evaluate::Type<
                       Fortran::common::TypeCategory::Character, KIND>> &x) {
  return mangleArrayLiteral(
      reinterpret_cast<const uint8_t *>(x.values().data()),
      x.values().size() * sizeof(x.values()[0]), x.shape(),
      Fortran::common::TypeCategory::Character, KIND, x.LEN());
}

// FIXME: derived type mangling is safe but not reproducible between two
// compilation of a same file because `values().data()` is a nontrivial compile
// time data structure containing pointers and vectors. In particular, this
// means that similar structure constructors are not "combined" into the same
// global constant by lowering.
inline std::string mangleArrayLiteral(
    mlir::Type eleTy,
    const Fortran::evaluate::Constant<Fortran::evaluate::SomeDerived> &x) {
  return mangleArrayLiteral(
      reinterpret_cast<const uint8_t *>(x.values().data()),
      x.values().size() * sizeof(x.values()[0]), x.shape(),
      Fortran::common::TypeCategory::Derived, /*kind=*/0, /*charLen=*/-1,
      eleTy.cast<fir::RecordType>().getName());
}

/// Return the compiler-generated name of a static namelist variable descriptor.
std::string globalNamelistDescriptorName(const Fortran::semantics::Symbol &sym);

} // namespace lower::mangle
} // namespace Fortran

#endif // FORTRAN_LOWER_MANGLER_H
