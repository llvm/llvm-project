//===-- include/flang/Parser/unparse.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_PARSER_UNPARSE_H_
#define FORTRAN_PARSER_UNPARSE_H_

#include "char-block.h"
#include "characters.h"
#include <functional>
#include <iosfwd>

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate {
struct GenericExprWrapper;
struct GenericAssignmentWrapper;
class ProcedureRef;
} // namespace Fortran::evaluate

namespace Fortran::parser {

struct Program;
struct Expr;

// A function called before each Statement is unparsed.
using preStatementType =
    std::function<void(const CharBlock &, llvm::raw_ostream &, int)>;

// Functions to handle unparsing of analyzed expressions and related
// objects rather than their original parse trees.
struct AnalyzedObjectsAsFortran {
  std::function<void(llvm::raw_ostream &, const evaluate::GenericExprWrapper &)>
      expr;
  std::function<void(
      llvm::raw_ostream &, const evaluate::GenericAssignmentWrapper &)>
      assignment;
  std::function<void(llvm::raw_ostream &, const evaluate::ProcedureRef &)> call;
};

// Converts parsed program (or fragment) to out as Fortran.
template <typename A>
void Unparse(llvm::raw_ostream &out, const A &root,
    Encoding encoding = Encoding::UTF_8, bool capitalizeKeywords = true,
    bool backslashEscapes = true, preStatementType *preStatement = nullptr,
    AnalyzedObjectsAsFortran * = nullptr);

extern template void Unparse(llvm::raw_ostream &out, const Program &program,
    Encoding encoding, bool capitalizeKeywords, bool backslashEscapes,
    preStatementType *preStatement, AnalyzedObjectsAsFortran *);
extern template void Unparse(llvm::raw_ostream &out, const Expr &expr,
    Encoding encoding, bool capitalizeKeywords, bool backslashEscapes,
    preStatementType *preStatement, AnalyzedObjectsAsFortran *);
} // namespace Fortran::parser

#endif
