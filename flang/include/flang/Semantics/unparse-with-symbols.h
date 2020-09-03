//===-- include/flang/Semantics/unparse-with-symbols.h ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_
#define FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_

#include "flang/Parser/characters.h"
#include <iosfwd>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {
struct Program;
}

namespace Fortran::semantics {
void UnparseWithSymbols(llvm::raw_ostream &, const parser::Program &,
    parser::Encoding encoding = parser::Encoding::UTF_8);
}

#endif // FORTRAN_SEMANTICS_UNPARSE_WITH_SYMBOLS_H_
