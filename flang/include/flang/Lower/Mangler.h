//===-- Lower/Mangler.h -- name mangling ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_MANGLER_H_
#define FORTRAN_LOWER_MANGLER_H_

#include <string>

namespace fir {
struct NameUniquer;
}

namespace llvm {
class StringRef;
}

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

#endif // FORTRAN_LOWER_MANGLER_H_
