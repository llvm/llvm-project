//===-- lib/Semantics/resolve-names.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_RESOLVE_NAMES_H_
#define FORTRAN_SEMANTICS_RESOLVE_NAMES_H_

#include <iosfwd>
#include <string>
#include <vector>

namespace Fortran::parser {
struct Program;
}

namespace Fortran::semantics {

class SemanticsContext;

bool ResolveNames(SemanticsContext &, const parser::Program &);
void DumpSymbols(std::ostream &);

}

#endif  // FORTRAN_SEMANTICS_RESOLVE_NAMES_H_
