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

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/StringRef.h"
#include <string>

namespace fir {
struct NameUniquer;

/// Returns a name suitable to define mlir functions for Fortran intrinsic
/// Procedure. These names are guaranteed to not conflict with user defined
/// procedures. This is needed to implement Fortran generic intrinsics as
/// several mlir functions specialized for the argument types.
/// The result is guaranteed to be distinct for different mlir::FunctionType
/// arguments. The mangling pattern is:
///    fir.<generic name>.<result type>.<arg type>...
/// e.g ACOS(COMPLEX(4)) is mangled as fir.acos.z4.z4
std::string mangleIntrinsicProcedure(llvm::StringRef genericName,
                                     mlir::FunctionType);
} // namespace fir

namespace Fortran {
namespace common {
template <typename>
class Reference;
}

namespace semantics {
class Symbol;
}

namespace lower {
namespace mangle {

/// Convert a front-end Symbol to an internal name
std::string mangleName(fir::NameUniquer &uniquer, const semantics::Symbol &);

std::string demangleName(llvm::StringRef name);

} // namespace mangle
} // namespace lower
} // namespace Fortran

#endif // FORTRAN_LOWER_MANGLER_H
